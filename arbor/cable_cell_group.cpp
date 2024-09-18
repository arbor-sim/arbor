#include <variant>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/sampling.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike.hpp>

#include "backends/event.hpp"
#include "cell_group.hpp"
#include "fvm_lowered_cell.hpp"
#include "label_resolution.hpp"
#include "cable_cell_group.hpp"
#include "profile/profiler_macro.hpp"
#include "sampler_map.hpp"
#include "util/partition.hpp"
#include "util/range.hpp"
#include "util/span.hpp"

namespace arb {

cable_cell_group::cable_cell_group(const std::vector<cell_gid_type>& gids,
                                   const recipe& rec,
                                   cell_label_range& cg_sources,
                                   cell_label_range& cg_targets,
                                   fvm_lowered_cell_ptr lowered):
    gids_(gids), lowered_(std::move(lowered))
{

    // Construct cell implementation, retrieving handles and maps.
    auto fvm_info = lowered_->initialize(gids_, rec);

    // Propagate source and target ranges to the simulator object
    cg_sources = std::move(fvm_info.source_data);
    cg_targets = std::move(fvm_info.target_data);

    probe_map_ = std::move(fvm_info.probe_map);

    // Create a list of the global identifiers for the spike sources
    for (auto source_gid: gids_) {
        for (cell_lid_type lid = 0; lid<fvm_info.num_sources[source_gid]; ++lid) {
            spike_sources_.push_back({source_gid, lid});
        }
    }
    spike_sources_.shrink_to_fit();
}

void cable_cell_group::reset() {
    spikes_.clear();

    for (auto &entry: sampler_map_) {
        entry.second.sched.reset();
    }

    lowered_->reset();
}

// Probe-type specific sample data marshalling.

struct sampler_call_info {
    sampler_function sampler;
    cell_address_type probeset_id;
    unsigned index;
    const fvm_probe_data* pdata_ptr;

    // Offsets are into lowered cell sample time and event arrays.
    sample_size_type begin_offset;
    sample_size_type end_offset;
};

void cable_cell_group::t_serialize(serializer& ser,
                                   const std::string& k) const { serialize(ser, k, *this); }
void cable_cell_group::t_deserialize(serializer& ser,
                                     const std::string& k) { deserialize(ser, k, *this); }

// Working space for computing and collating data for samplers.
using fvm_probe_scratch = std::tuple<std::vector<double>, std::vector<cable_sample_range>>;

template <typename VoidFn, typename... A>
void tuple_foreach(VoidFn&& f, std::tuple<A...>& t) {
    // executes functions in order (pack expansion)
    // uses comma operator (unary left fold)
    // avoids potentially overloaded comma operator (cast to void)
    std::apply(
        [g = std::forward<VoidFn>(f)](auto&& ...x){
            (..., static_cast<void>(g(std::forward<decltype(x)>(x))));},
        t);
}

void reserve_scratch(fvm_probe_scratch& scratch, std::size_t n) {
    tuple_foreach([n](auto& v) { v.reserve(n); }, scratch);
}

void run_samples(
    const missing_probe_info&,
    const sampler_call_info&,
    const arb_value_type*,
    const arb_value_type*,
    std::vector<sample_record>&,
    fvm_probe_scratch&)
{
    throw arbor_internal_error("invalid fvm_probe_data in sampler map");
}

void run_samples(
    const fvm_probe_scalar& p,
    const sampler_call_info& sc,
    const arb_value_type* raw_times,
    const arb_value_type* raw_samples,
    std::vector<sample_record>& sample_records,
    fvm_probe_scratch&)
{
    // Scalar probes do not need scratch space — provided that the user-presented
    // sample type (double) matches the raw type (arb_value_type).
    static_assert(std::is_same<double, arb_value_type>::value, "require sample value translation");

    sample_size_type n_sample = sc.end_offset-sc.begin_offset;
    sample_records.clear();
    for (auto i = sc.begin_offset; i!=sc.end_offset; ++i) {
       sample_records.push_back(sample_record{time_type(raw_times[i]), &raw_samples[i]});
    }

    sc.sampler({sc.probeset_id, sc.index, p.get_metadata_ptr()}, n_sample, sample_records.data());
}

void run_samples(
    const fvm_probe_interpolated& p,
    const sampler_call_info& sc,
    const arb_value_type* raw_times,
    const arb_value_type* raw_samples,
    std::vector<sample_record>& sample_records,
    fvm_probe_scratch& scratch)
{
    constexpr sample_size_type n_raw_per_sample = 2;
    sample_size_type n_sample = (sc.end_offset-sc.begin_offset)/n_raw_per_sample;
    arb_assert((sc.end_offset-sc.begin_offset)==n_sample*n_raw_per_sample);

    auto& tmp = std::get<std::vector<double>>(scratch);
    tmp.clear();
    sample_records.clear();

    for (sample_size_type j = 0; j<n_sample; ++j) {
        auto offset = j*n_raw_per_sample+sc.begin_offset;
        tmp.push_back(p.coef[0]*raw_samples[offset] + p.coef[1]*raw_samples[offset+1]);
    }

    const auto& ctmp = tmp;
    for (sample_size_type j = 0; j<n_sample; ++j) {
        auto offset = j*n_raw_per_sample+sc.begin_offset;
        sample_records.push_back(sample_record{time_type(raw_times[offset]), &ctmp[j]});
    }

    sc.sampler({sc.probeset_id, sc.index, p.get_metadata_ptr()}, n_sample, sample_records.data());
}

void run_samples(
    const fvm_probe_multi& p,
    const sampler_call_info& sc,
    const arb_value_type* raw_times,
    const arb_value_type* raw_samples,
    std::vector<sample_record>& sample_records,
    fvm_probe_scratch& scratch)
{
    const sample_size_type n_raw_per_sample = p.raw_handles.size();
    sample_size_type n_sample = (sc.end_offset-sc.begin_offset)/n_raw_per_sample;
    arb_assert((sc.end_offset-sc.begin_offset)==n_sample*n_raw_per_sample);

    auto& sample_ranges = std::get<std::vector<cable_sample_range>>(scratch);
    sample_ranges.clear();
    sample_records.clear();

    for (sample_size_type j = 0; j<n_sample; ++j) {
        auto offset = j*n_raw_per_sample+sc.begin_offset;
        sample_ranges.push_back({raw_samples+offset, raw_samples+offset+n_raw_per_sample});
    }

    const auto& csample_ranges = sample_ranges;
    for (sample_size_type j = 0; j<n_sample; ++j) {
        auto offset = j*n_raw_per_sample+sc.begin_offset;
        sample_records.push_back(sample_record{time_type(raw_times[offset]), &csample_ranges[j]});
    }

    sc.sampler({sc.probeset_id, sc.index, p.get_metadata_ptr()}, n_sample, sample_records.data());
}

void run_samples(
    const fvm_probe_weighted_multi& p,
    const sampler_call_info& sc,
    const arb_value_type* raw_times,
    const arb_value_type* raw_samples,
    std::vector<sample_record>& sample_records,
    fvm_probe_scratch& scratch)
{
    const sample_size_type n_raw_per_sample = p.raw_handles.size();
    sample_size_type n_sample = (sc.end_offset - sc.begin_offset)/n_raw_per_sample;
    arb_assert((sc.end_offset - sc.begin_offset)==n_sample*n_raw_per_sample);
    arb_assert((unsigned)n_raw_per_sample == p.weight.size());

    auto& sample_ranges = std::get<std::vector<cable_sample_range>>(scratch);
    sample_ranges.clear();
    sample_records.clear();

    auto& tmp = std::get<std::vector<double>>(scratch);
    tmp.clear();
    tmp.reserve(n_raw_per_sample*n_sample);

    for (sample_size_type j = 0; j<n_sample; ++j) {
        auto offset = j*n_raw_per_sample+sc.begin_offset;
        for (sample_size_type i = 0; i<n_raw_per_sample; ++i) {
            tmp.push_back(raw_samples[offset+i]*p.weight[i]);
        }
    }

    const double* tmp_ptr = tmp.data();
    for (sample_size_type j = 0; j<n_sample; ++j) {
        sample_ranges.push_back({tmp_ptr, tmp_ptr+n_raw_per_sample});
        tmp_ptr += n_raw_per_sample;
    }

    const auto& csample_ranges = sample_ranges;
    for (sample_size_type j = 0; j<n_sample; ++j) {
        auto offset = j*n_raw_per_sample+sc.begin_offset;
        sample_records.push_back(sample_record{time_type(raw_times[offset]), &csample_ranges[j]});
    }

    sc.sampler({sc.probeset_id, sc.index, p.get_metadata_ptr()}, n_sample, sample_records.data());
}

void run_samples(const fvm_probe_interpolated_multi& p,
                 const sampler_call_info& sc,
                 const arb_value_type* raw_times,
                 const arb_value_type* raw_samples,
                 std::vector<sample_record>& sample_records,
                 fvm_probe_scratch& scratch) {
    const sample_size_type n_raw_per_sample = p.raw_handles.size();
    const sample_size_type n_interp_per_sample = n_raw_per_sample/2;
    sample_size_type n_sample = (sc.end_offset-sc.begin_offset)/n_raw_per_sample;
    arb_assert((sc.end_offset-sc.begin_offset)==n_sample*n_raw_per_sample);
    arb_assert((unsigned)n_interp_per_sample==p.coef[0].size());
    arb_assert((unsigned)n_interp_per_sample==p.coef[1].size());

    auto& sample_ranges = std::get<std::vector<cable_sample_range>>(scratch);
    sample_ranges.clear();
    sample_records.clear();

    auto& tmp = std::get<std::vector<double>>(scratch);
    tmp.clear();
    tmp.reserve(n_interp_per_sample*n_sample);

    for (sample_size_type j = 0; j<n_sample; ++j) {
        auto offset = j*n_raw_per_sample+sc.begin_offset;
        const auto* raw_a = raw_samples + offset;
        const auto* raw_b = raw_a + n_interp_per_sample;
        for (sample_size_type i = 0; i<n_interp_per_sample; ++i) {
            tmp.push_back(raw_a[i]*p.coef[0][i]+raw_b[i]*p.coef[1][i]);
        }
    }

    const double* tmp_ptr = tmp.data();
    for (sample_size_type j = 0; j<n_sample; ++j) {
        sample_ranges.push_back({tmp_ptr, tmp_ptr+n_interp_per_sample});
        tmp_ptr += n_interp_per_sample;
    }

    const auto& csample_ranges = sample_ranges;
    for (sample_size_type j = 0; j<n_sample; ++j) {
        auto offset = j*n_interp_per_sample+sc.begin_offset;
        sample_records.push_back(sample_record{time_type(raw_times[offset]), &csample_ranges[j]});
    }

    sc.sampler({sc.probeset_id, sc.index, p.get_metadata_ptr()}, n_sample, sample_records.data());
}

void run_samples(
    const fvm_probe_membrane_currents& p,
    const sampler_call_info& sc,
    const arb_value_type* raw_times,
    const arb_value_type* raw_samples,
    std::vector<sample_record>& sample_records,
    fvm_probe_scratch& scratch)
{
    const sample_size_type n_raw_per_sample = p.raw_handles.size();
    sample_size_type n_sample = (sc.end_offset-sc.begin_offset)/n_raw_per_sample;
    arb_assert((sc.end_offset-sc.begin_offset)==n_sample*n_raw_per_sample);

    const auto n_cable = p.metadata.size();
    const auto n_cv = p.cv_parent_cond.size();
    const auto cables_by_cv = util::partition_view(p.cv_cables_divs);
    const auto n_stim = p.stim_scale.size();
    arb_assert(n_stim+n_cv==(unsigned)n_raw_per_sample);

    auto& sample_ranges = std::get<std::vector<cable_sample_range>>(scratch);
    sample_ranges.clear();

    auto& tmp = std::get<std::vector<double>>(scratch);
    tmp.assign(n_cable*n_sample, 0.);

    sample_records.clear();

    for (sample_size_type j = 0; j<n_sample; ++j) {
        auto offset = j*n_raw_per_sample+sc.begin_offset;
        auto tmp_base = tmp.data()+j*n_cable;

        // Each CV voltage contributes to the current sum of its parent's cables
        // and its own cables.

        const double* v = raw_samples+offset;
        for (auto cv: util::make_span(n_cv)) {
            arb_index_type parent_cv = p.cv_parent[cv];
            if (parent_cv+1==0) continue;

            double cond = p.cv_parent_cond[cv];

            double cv_I = v[cv]*cond;
            double parent_cv_I = v[parent_cv]*cond;

            for (auto cable_i: util::make_span(cables_by_cv[cv])) {
                tmp_base[cable_i] -= (cv_I-parent_cv_I)*p.weight[cable_i];
            }

            for (auto cable_i: util::make_span(cables_by_cv[parent_cv])) {
                tmp_base[cable_i] += (cv_I-parent_cv_I)*p.weight[cable_i];
            }
        }

        const double* stim = raw_samples+offset+n_cv;
        for (auto i: util::make_span(n_stim)) {
            double cv_stim_I = stim[i]*p.stim_scale[i];
            unsigned cv = p.stim_cv[i];
            arb_assert(cv<n_cv);

            for (auto cable_i: util::make_span(cables_by_cv[cv])) {
                tmp_base[cable_i] -= cv_stim_I*p.weight[cable_i];
            }
        }
        sample_ranges.push_back({tmp_base, tmp_base+n_cable});
    }

    const auto& csample_ranges = sample_ranges;
    for (sample_size_type j = 0; j<n_sample; ++j) {
        auto offset = j*n_raw_per_sample+sc.begin_offset;
        sample_records.push_back(sample_record{time_type(raw_times[offset]), &csample_ranges[j]});
    }

    sc.sampler({sc.probeset_id, sc.index, p.get_metadata_ptr()}, n_sample, sample_records.data());
}

// Generic run_samples dispatches on probe info variant type.
void run_samples(
    const sampler_call_info& sc,
    const arb_value_type* raw_times,
    const arb_value_type* raw_samples,
    std::vector<sample_record>& sample_records,
    fvm_probe_scratch& scratch)
{
    std::visit([&](auto& x) {run_samples(x, sc, raw_times, raw_samples, sample_records, scratch); }, sc.pdata_ptr->info);
}

void cable_cell_group::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    time_type tstart = lowered_->time();

    // Bin and collate deliverable events from event lanes.
    // Split epoch into equally sized timesteps (last timestep is chosen to match end of epoch)
    timesteps_.reset(ep, dt);

    PE(advance:samplesetup:clear);
    sample_events_.resize(timesteps_.size());
    for (auto& v: sample_events_) v.clear();
    PL();

    // Create sample events and delivery information.
    //
    // For each (schedule, sampler, probe set) in the sampler association
    // map that will be triggered in this integration interval, create
    // sample events for the lowered cell, one or more for each scheduled
    // sample time and probe in the probe set.
    //
    // Each event is associated with an offset into the sample data and
    // time buffers; these are assigned contiguously such that one call to
    // a sampler callback can be represented by a `sampler_call_info`
    // value as defined below, grouping together all the samples of the
    // same probe for this callback in this association.

    PE(advance:samplesetup);
    std::vector<sampler_call_info> call_info;

    sample_size_type n_samples = 0;
    sample_size_type max_samples_per_call = 0;

    if (!sampler_map_.empty()) { // NOTE: We avoid the lock here as often as possible
        // SAFETY: We need the lock here, as _schedule_ is not reentrant.
        std::lock_guard<std::mutex> guard(sampler_mex_);
        for (auto& [sk, sa]: sampler_map_) {
            if (sa.probeset_ids.empty()) continue; // No need to make any schedule
            auto sample_times = util::make_range(sa.sched.events(tstart, ep.t1));
            sample_size_type n_times = sample_times.size();
            if (n_times == 0) continue;
            max_samples_per_call = std::max(max_samples_per_call, n_times);
            for (const auto& pid: sa.probeset_ids) {
                unsigned index = 0;
                for (const auto& pdata: probe_map_.data_on(pid)) {
                    call_info.push_back({sa.sampler,
                                         pid,
                                         index,
                                         pdata,
                                         n_samples,
                                         n_samples + n_times*pdata->n_raw()});
                    index++;
                    for (auto t: sample_times) {
                        auto it = timesteps_.find(t);
                        arb_assert(it != timesteps_.end());
                        const auto timestep_index = it - timesteps_.begin();
                        for (probe_handle h: pdata->raw_handle_range()) {
                            sample_event ev{t, {h, n_samples++}};
                            sample_events_[timestep_index].push_back(ev);
                        }
                    }
                }
            }
            arb_assert(n_samples==call_info.back().end_offset);
        }
    }
    PL();

    // Run integration and collect samples, spikes.
    auto result = lowered_->integrate(timesteps_, event_lanes, sample_events_);

    // For each sampler callback registered in `call_info`, construct the
    // vector of sample entries from the lowered cell sample times and values
    // and then call the callback.

    PE(advance:sampledeliver);
    std::vector<sample_record> sample_records;
    sample_records.reserve(max_samples_per_call);

    fvm_probe_scratch scratch;
    reserve_scratch(scratch, max_samples_per_call);

    for (auto& sc: call_info) {
        run_samples(sc, result.sample_time.data(), result.sample_value.data(), sample_records, scratch);
    }
    PL();

    // Copy out spike voltage threshold crossings from the back end, then
    // generate spikes with global spike source ids. The threshold crossings
    // record the local spike source index, which must be converted to a
    // global index for spike communication.

    for (auto c: result.crossings) {
        spikes_.emplace_back(spike_sources_[c.index], time_type(c.time));
    }
}

void cable_cell_group::add_sampler(sampler_association_handle h,
                                   cell_member_predicate probeset_ids,
                                   schedule sched,
                                   sampler_function fn) {
    // SAFETY? Both probe_map and sampler must be protected by this lock?!
    std::lock_guard<std::mutex> guard(sampler_mex_);
    auto probeset = probe_map_.keys(probeset_ids);
    if (!probeset.empty()) {
        auto result = sampler_map_.insert({h, sampler_association{std::move(sched),
                                                                  std::move(fn),
                                                                  std::move(probeset)}});
        arb_assert(result.second);
    }
}

void cable_cell_group::remove_sampler(sampler_association_handle h) {
    std::lock_guard<std::mutex> guard(sampler_mex_);
    sampler_map_.erase(h);
}

void cable_cell_group::remove_all_samplers() {
    std::lock_guard<std::mutex> guard(sampler_mex_);
    sampler_map_.clear();
}

std::vector<probe_metadata> cable_cell_group::get_probe_metadata(const cell_address_type& probeset_id) const {
    // SAFETY: Probe associations are fixed after construction, so we do not
    //         need to grab the mutex.
    auto data = probe_map_.data_on(probeset_id);

    std::vector<probe_metadata> result;
    result.reserve(data.size());
    unsigned index = 0;
    for (const auto& info: data) {
        result.push_back({probeset_id, index++, info->get_metadata_ptr()});
    }
    return result;
}
} // namespace arb
