#include <exception>

#include <cell_group.hpp>
#include "profile/profiler_macro.hpp"
#include <recipe.hpp>
#include <spike_source_cell.hpp>
#include <spike_source_cell_group.hpp>
#include <time_sequence.hpp>

namespace arb {

spike_source_cell_group::spike_source_cell_group(std::vector<cell_gid_type> gids, const recipe& rec):
    gids_(std::move(gids))
{
    time_sequences_.reserve(gids_.size());
    for (auto gid: gids_) {
        try {
            auto cell = util::any_cast<spike_source_cell>(rec.get_cell_description(gid));
            time_sequences_.push_back(std::move(cell.seq));
        }
        catch (util::bad_any_cast& e) {
            throw std::runtime_error("model cell type mismatch: gid "+std::to_string(gid)+" is not a spike_source_cell");
        }
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

        while (tseq.front()<ep.tfinal) {
            spikes_.push_back({{gid, 0u}, tseq.front()});
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

void spike_source_cell_group::add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) {
    std::logic_error("A spike_source_cell group doen't support sampling of internal state!");
}

} // namespace arb


