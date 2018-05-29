#pragma once

#include <cell_group.hpp>
#include <recipe.hpp>
#include <time_sequence.hpp>
#include <profiling/profiler.hpp>

namespace arb {

class proxy_group: public cell_group {
public:
    proxy_group(std::vector<cell_gid_type> gids, const recipe& rec):
        gids_(std::move(gids))
    {
        time_sequences_.reserve(gids_.size());
        for (auto gid: gids_) {
            time_sequences_.push_back(util::any_cast<time_seq>(rec.get_cell_description(gid)));
        }
    }

    cell_kind get_cell_kind() const override {
        return cell_kind::proxy_spike_source;
    }

    void advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) override {
        PE(advance_proxy);

        for (auto i: util::make_span(0, gids_.size())) {
            auto& tseq = time_sequences_[i];
            const auto gid = gids_[i];

            while (tseq.next()<ep.tfinal) {
                spikes_.push_back({{gids_[i], 0u}, tseq.next()});
            }
        }
        PL();
    };

    void reset() override {
        for (auto& s: time_sequences_) {
            s.reset();
        }

        clear_spikes();
    }

    const std::vector<spike>& spikes() const override {
        return spikes_;
    }

    void clear_spikes() override {
        spikes_.clear();
    }

    void add_sampler(sampler_association_handle h, cell_member_predicate probe_ids, schedule sched, sampler_function fn, sampling_policy policy) override {
        std::logic_error("A proxy_cell group doen't support sampling of internal state!");
    }

    void remove_sampler(sampler_association_handle h) override {}

    void remove_all_samplers() override {}

private:
    std::vector<spike> spikes_;
    std::vector<cell_gid_type> gids_;
    std::vector<time_seq> time_sequences_;
};

class bench_cell {
public:
    bench_cell(time_seq times): times_(std::move(times)) {}

private:
    time_seq times_;
};

} // namespace arb
