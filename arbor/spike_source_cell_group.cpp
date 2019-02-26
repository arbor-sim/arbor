#include <exception>

#include <arbor/recipe.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/schedule.hpp>

#include "cell_group.hpp"
#include "profile/profiler_macro.hpp"
#include "spike_source_cell_group.hpp"
#include "util/span.hpp"

namespace arb {

spike_source_cell_group::spike_source_cell_group(const std::vector<cell_gid_type>& gids, const recipe& rec):
    gids_(gids)
{
    time_sequences_.reserve(gids_.size());
    for (auto gid: gids_) {
        try {
            auto cell = util::any_cast<spike_source_cell>(rec.get_cell_description(gid));
            time_sequences_.push_back(std::move(cell.seq));
        }
        catch (util::bad_any_cast& e) {
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

        for (auto t: util::make_range(time_sequences_[i].events(t_, ep.tfinal))) {
            spikes_.push_back({{gid, 0u}, t});
        }
    }
    t_ = ep.tfinal;

    PL();
};

void spike_source_cell_group::reset() {
    for (auto& s: time_sequences_) {
        s.reset();
    }
    t_ = 0;

    clear_spikes();
}

const std::vector<spike>& spike_source_cell_group::spikes() const {
    return spikes_;
}

void spike_source_cell_group::clear_spikes() {
    spikes_.clear();
}

void spike_source_cell_group::add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) {
    std::logic_error("A spike_source_cell group doen't support sampling of internal state!");
}

} // namespace arb


