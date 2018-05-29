#include <exception>

#include <cell_group.hpp>
#include <recipe.hpp>
#include <time_sequence.hpp>
#include <profiling/profiler.hpp>
#include <spike_source_cell_group.hpp>

namespace arb {

spike_source_cell_group::spike_source_cell_group(std::vector<cell_gid_type> gids, const recipe& rec):
    gids_(std::move(gids))
{
    time_sequences_.reserve(gids_.size());
    for (auto gid: gids_) {
        time_sequences_.push_back(util::any_cast<time_seq>(rec.get_cell_description(gid)));
    }
}

cell_kind spike_source_cell_group::get_cell_kind() const {
    return cell_kind::spike_source;
}

void spike_source_cell_group::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    PE(advance_sscell);

    for (auto i: util::make_span(0, gids_.size())) {
        auto& tseq = time_sequences_[i];
        const auto gid = gids_[i];

        while (tseq.next()<ep.tfinal) {
            spikes_.push_back({{gid, 0u}, tseq.next()});
            tseq.pop();
        }
    }
    PL();
};

void spike_source_cell_group::reset() {
    for (auto& s: time_sequences_) {
        s.reset();
    }

    clear_spikes();
}

const std::vector<spike>& spike_source_cell_group::spikes() const {
    return spikes_;
}

void spike_source_cell_group::clear_spikes() {
    spikes_.clear();
}

void spike_source_cell_group::add_sampler(sampler_association_handle h,
                                   cell_member_predicate probe_ids,
                                   schedule sched,
                                   sampler_function fn,
                                   sampling_policy policy)
{
    std::logic_error("A spike_source_cell group doen't support sampling of internal state!");
}

void spike_source_cell_group::remove_sampler(sampler_association_handle h) {}

void spike_source_cell_group::remove_all_samplers() {}

void spike_source_cell_group::set_binning_policy(binning_kind policy, time_type bin_interval) {}

} // namespace arb


