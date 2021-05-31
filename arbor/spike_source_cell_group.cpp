#include <exception>

#include <arbor/arbexcept.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/schedule.hpp>

#include "cell_group.hpp"
#include "label_resolution.hpp"
#include "profile/profiler_macro.hpp"
#include "spike_source_cell_group.hpp"
#include "util/span.hpp"

namespace arb {

spike_source_cell_group::spike_source_cell_group(
    const std::vector<cell_gid_type>& gids,
    const recipe& rec,
    cell_label_range& cg_sources,
    cell_label_range& cg_targets):
    gids_(gids)
{
    for (auto gid: gids_) {
        if (!rec.get_probes(gid).empty()) {
            throw bad_cell_probe(cell_kind::spike_source, gid);
        }
    }

    time_sequences_.reserve(gids_.size());
    for (auto gid: gids_) {
        cg_sources.add_cell();
        cg_targets.add_cell();
        try {
            auto cell = util::any_cast<spike_source_cell>(rec.get_cell_description(gid));
            time_sequences_.push_back(std::move(cell.seq));
            cg_sources.add_label(cell.source, {0, 1});
        }
        catch (std::bad_any_cast& e) {
            throw bad_cell_description(cell_kind::spike_source, gid);
        }
    }
}

cell_kind spike_source_cell_group::get_cell_kind() const {
    return cell_kind::spike_source;
}

void spike_source_cell_group::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    PE(advance_sscell);

    for (auto i: util::count_along(gids_)) {
        const auto gid = gids_[i];

        for (auto t: util::make_range(time_sequences_[i].events(ep.t0, ep.t1))) {
            spikes_.push_back({{gid, 0u}, t});
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

void spike_source_cell_group::add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) {
    throw std::logic_error("A spike_source_cell group doen't support sampling of internal state!");
}

} // namespace arb


