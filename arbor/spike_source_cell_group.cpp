#include <arbor/arbexcept.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/schedule.hpp>

#include "cell_group.hpp"
#include "label_resolution.hpp"
#include "profile/profiler_macro.hpp"
#include "spike_source_cell_group.hpp"
#include "util/span.hpp"
#include "util/maputil.hpp"

namespace arb {

spike_source_cell_group::spike_source_cell_group(const std::vector<cell_gid_type>& gids,
                                                 const recipe& rec,
                                                 cell_label_range& cg_sources,
                                                 cell_label_range& cg_targets):
    gids_(gids) {
    for (auto gid: gids_) {
        if (!rec.get_probes(gid).empty()) {
            throw bad_cell_probe(cell_kind::spike_source, gid);
        }
    }

    time_sequences_.reserve(gids.size());
    for (auto gid: gids_) {
        cg_sources.add_cell();
        cg_targets.add_cell();
        try {
            auto cell = util::any_cast<spike_source_cell>(rec.get_cell_description(gid));
            time_sequences_.emplace_back(cell.schedules);
            cg_sources.add_label(hash_value(cell.source), {0, 1});
            sources_.push_back(cell.source);
        }
        catch (std::bad_any_cast& e) {
            throw bad_cell_description(cell_kind::spike_source, gid);
        }
    }
}

cell_kind spike_source_cell_group::get_cell_kind() const { return cell_kind::spike_source; }

void spike_source_cell_group::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    PE(advance:sscell);

    for (auto i: util::count_along(gids_)) {
        const auto gid = gids_[i];

        for (auto& ts: time_sequences_[i]) {
            for (auto &t: util::make_range(ts.events(ep.t0, ep.t1))) {
                spikes_.push_back({{gid, 0u}, t});
            }
        }
    }

    PL();
};

void spike_source_cell_group::reset() {
    for (auto& ss: time_sequences_) {
        for(auto& s: ss) {
            s.reset();
        }
    }
    clear_spikes();
}

void
spike_source_cell_group::edit_cell(cell_gid_type gid, std::any cell_edit) {
    try {
        auto source_edit = std::any_cast<spike_source_cell_editor>(cell_edit);
        auto lid = util::binary_search_index(gids_, gid);
        if (!lid) throw arb::arbor_internal_error{"gid " + std::to_string(gid) + " erroneuosly dispatched to cell group."};
        auto idx = *lid;
        auto tmp = spike_source_cell{sources_[idx], std::move(time_sequences_[idx])};
        source_edit(tmp);
        // NOTE: we forbid writing to V_m? Reasons
        //       * the cell might be in the refractory period which causes semantic issues
        //         - return to normal or not?
        //         - what should probes return
        //       * V_m is the _initial state_ only
        if (tmp.source != sources_[idx]) throw bad_cell_edit(gid, "Source is not editable.");
        // Write back
        time_sequences_[idx] = std::move(tmp.schedules);
    }
    catch (const std::bad_any_cast&) {
        throw bad_cell_edit(gid, "Not a source cell editor (C++ typid: '" + std::string{cell_edit.type().name()} + "')");
    }
}

const std::vector<spike>& spike_source_cell_group::spikes() const { return spikes_; }

void spike_source_cell_group::clear_spikes() { spikes_.clear(); }

void spike_source_cell_group::t_serialize(serializer& ser, const std::string& k) const { serialize(ser, k, *this); }

void spike_source_cell_group::t_deserialize(serializer& ser, const std::string& k) { deserialize(ser, k, *this); }

void spike_source_cell_group::add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function) {}

} // namespace arb


