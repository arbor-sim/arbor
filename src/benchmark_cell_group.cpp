#include <chrono>
#include <exception>

#include <cell_group.hpp>
#include <profiling/profiler.hpp>
#include <recipe.hpp>
#include <benchmark_cell.hpp>
#include <benchmark_cell_group.hpp>
#include <time_sequence.hpp>

namespace arb {

benchmark_cell_group::benchmark_cell_group(std::vector<cell_gid_type> gids,
                                           const recipe& rec):
    gids_(std::move(gids))
{
    cells_.reserve(gids_.size());
    for (auto gid: gids_) {
        cells_.push_back(util::any_cast<benchmark_cell>(rec.get_cell_description(gid)));
    }

    reset();
}

void benchmark_cell_group::reset() {
    t_ = 0;

    for (auto& c: cells_) {
        c.time_sequence.reset();
    }

    clear_spikes();
}

cell_kind benchmark_cell_group::get_cell_kind() const {
    return cell_kind::benchmark;
}

void benchmark_cell_group::advance(epoch ep,
                                   time_type dt,
                                   const event_lane_subrange& event_lanes)
{
    using std::chrono::high_resolution_clock;
    using duration_type = std::chrono::duration<double, std::micro>;

    PE(advance_bench_cell);
    for (auto i: util::make_span(0, gids_.size())) {
        auto& tseq = cells_[i].time_sequence;
        // expected time to complete epoch in micro seconds.
        const double duration_us = cells_[i].run_time_per_ms*(ep.tfinal-t_);
        const auto gid = gids_[i];

        // start timer
        auto start = high_resolution_clock::now();

        while (tseq.front()<ep.tfinal) {
            spikes_.push_back({{gid, 0u}, tseq.front()});
            tseq.pop();
        }

        // wait until the expected time to advance has elapsed.
        while(duration_type(high_resolution_clock::now()-start).count() < duration_us);
    }
    t_ = ep.tfinal;

    PL();
};

const std::vector<spike>& benchmark_cell_group::spikes() const {
    return spikes_;
}

void benchmark_cell_group::clear_spikes() {
    spikes_.clear();
}

void benchmark_cell_group::add_sampler(sampler_association_handle h,
                                   cell_member_predicate probe_ids,
                                   schedule sched,
                                   sampler_function fn,
                                   sampling_policy policy)
{
    std::logic_error("A benchmark_cell group doen't support sampling of internal state!");
}

void benchmark_cell_group::remove_sampler(sampler_association_handle h) {}

void benchmark_cell_group::remove_all_samplers() {}

void benchmark_cell_group::set_binning_policy(binning_kind policy, time_type bin_interval) {}

} // namespace arb
