#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include <arbor/arbexcept.hpp>

#include "communication/neighborhood_router.hpp"
#include "profile/profiler_macro.hpp"
#include "util/rangeutil.hpp"

namespace arb {

std::size_t exchange_period(time_type delay, time_type interval) {
    arb_assert(interval > 0);
    // Leave some slack for rounding in the epoch boundaries; this only matters
    // if delay is a multiple of interval and then costs a single epoch.
    auto ratio = std::floor(delay/interval*(1 - 1e-6));
    ratio = std::min(ratio, time_type(std::numeric_limits<int>::max()));
    return ratio > 2 ? std::size_t(ratio) - 1 : 1;
}

void merge_sorted_runs(std::vector<spike>& spikes, std::vector<std::size_t> runs) {
    auto by_source = [](const spike& a, const spike& b) { return a.source < b.source; };
    // Boundaries of the runs; merge neighbouring pairs until a single run is left.
    runs.push_back(spikes.size());
    while (runs.size() > 2) {
        std::vector<std::size_t> merged{runs.front()};
        for (std::size_t i = 0; i + 2 < runs.size(); i += 2) {
            std::inplace_merge(spikes.begin() + runs[i], spikes.begin() + runs[i + 1], spikes.begin() + runs[i + 2], by_source);
            merged.push_back(runs[i + 2]);
        }
        // With an odd number of runs, the last one is merged in the next pass.
        if (runs.size() % 2 == 0) merged.push_back(runs.back());
        runs = std::move(merged);
    }
}

unsigned period_class(std::size_t period) {
    arb_assert(period > 0);
    unsigned c = 0;
    while (period >>= 1) ++c;
    return c;
}

namespace {
// Throw on all ranks if any rank fails a check, so that no rank is left
// waiting in a collective.
void check_on_all_ranks(const distributed_context& dist, bool ok, const std::string& what) {
    if (!dist.min(int(ok))) throw arbor_exception("neighborhood_spike_exchange: " + what);
}
} // namespace

neighborhood_router::neighborhood_router(const neighborhood_spike_exchange& options,
                                         distributed_context_handle dist):
    dist_(std::move(dist)),
    automatic_(options.automatic),
    rank_(dist_->id()),
    num_domains_(dist_->size())
{
    check_on_all_ranks(*dist_,
                       dist_->min(int(automatic_)) == dist_->max(int(automatic_)),
                       "all ranks must choose automatic levels alike.");
    if (automatic_) {
        check_on_all_ranks(*dist_, options.neighborhoods.empty(),
                           "neighborhoods must be empty with automatic levels.");
        // Until the connections are known, all other ranks are far ranks.
        set_levels({});
        return;
    }

    const int num_neighborhoods = options.neighborhoods.size();
    check_on_all_ranks(*dist_,
                       dist_->min(num_neighborhoods) == dist_->max(num_neighborhoods),
                       "all ranks must pass the same number of neighborhoods.");
    bool ranks_valid = true;
    for (const auto& neighborhood: options.neighborhoods) {
        for (auto r: neighborhood) ranks_valid &= r >= 0 && r < num_domains_;
    }
    check_on_all_ranks(*dist_, ranks_valid, "rank out of range.");
    set_levels(options.neighborhoods);
}

void neighborhood_router::set_levels(const std::vector<std::vector<int>>& neighborhoods) {
    const int num_neighborhoods = neighborhoods.size();
    levels_.clear();

    // Assign every other rank to the first neighbourhood that lists it, or the far level.
    const int far = num_neighborhoods;
    recv_level_.assign(num_domains_, far);
    recv_level_[rank_] = self_level;
    for (int l = 0; l < num_neighborhoods; ++l) {
        std::vector<int> sources;
        for (auto r: neighborhoods[l]) {
            if (recv_level_[r] != far) continue;
            recv_level_[r] = l;
            sources.push_back(r);
        }
        levels_.push_back({dist_->make_neighborhood(sources)});
    }
    levels_.emplace_back();

    // We are in exactly one level of each rank that receives from us.
    send_level_.assign(num_domains_, far);
    send_level_[rank_] = self_level;
    for (int l = 0; l < num_neighborhoods; ++l) {
        for (auto r: levels_[l].neighbors->destinations()) {
            arb_assert(send_level_[r] == far);
            send_level_[r] = l;
        }
    }
}

std::vector<std::vector<int>>
neighborhood_router::automatic_levels(const std::vector<time_type>& min_delays, time_type interval) const {
    // Class of each rank we receive from; -1 if there are no connections from it.
    std::vector<int> classes(num_domains_, -1);
    int max_class = -1;
    for (int d = 0; d < num_domains_; ++d) {
        if (d == rank_ || !std::isfinite(min_delays[d])) continue;
        classes[d] = period_class(exchange_period(min_delays[d], interval));
        max_class = std::max(max_class, classes[d]);
    }
    // All ranks need the same number of levels; the slowest class is the far level.
    max_class = dist_->max(max_class);
    std::vector<std::vector<int>> neighborhoods(std::max(max_class, 0));
    for (int d = 0; d < num_domains_; ++d) {
        if (classes[d] >= 0 && classes[d] < max_class) neighborhoods[classes[d]].push_back(d);
    }
    return neighborhoods;
}

void neighborhood_router::update(const std::vector<time_type>& min_delays,
                                 const source_ranks_map& src_ranks,
                                 time_type interval) {
    arb_assert(min_delays.size() == std::size_t(num_domains_));
    clear();
    if (automatic_) set_levels(automatic_levels(min_delays, interval));

    // All ranks must agree on when a level is exchanged, so periods use the global minimum delay.
    std::vector<time_type> level_delays(levels_.size(), std::numeric_limits<time_type>::infinity());
    for (int d = 0; d < num_domains_; ++d) {
        auto l = recv_level_[d];
        if (l != self_level) level_delays[l] = std::min(level_delays[l], min_delays[d]);
    }
    for (std::size_t l = 0; l < levels_.size(); ++l) {
        auto delay = dist_->min(level_delays[l]);
        levels_[l].period = std::isfinite(delay) ? exchange_period(delay, interval) : 0;
    }

    routes_.clear();
    for (const auto& [src, ranks]: src_ranks) {
        route r;
        for (auto d: ranks) {
            auto l = send_level_[d];
            if (l == self_level) continue;
            r.ranks.push_back(d);
            if (std::find(r.levels.begin(), r.levels.end(), l) == r.levels.end()) r.levels.push_back(l);
        }
        if (!r.levels.empty()) routes_.emplace(src, std::move(r));
    }
}

gathered_vector<spike>
neighborhood_router::exchange(const std::vector<spike>& local_spikes, std::optional<std::ptrdiff_t> epoch_id) {
    using count_type = gathered_vector<spike>::count_type;

    PE(buffer);
    arb_assert(std::is_sorted(local_spikes.begin(), local_spikes.end(),
                              [](const spike& a, const spike& b) { return a.source < b.source; }));
    // Every epoch appends a run sorted by source to each level; empty runs are dropped.
    for (auto& lvl: levels_) lvl.runs.push_back(lvl.pending.size());
    for (const auto& spk: local_spikes) {
        auto it = routes_.find(spk.source);
        if (it == routes_.end()) continue;
        for (auto l: it->second.levels) levels_[l].pending.push_back(spk);
    }
    for (auto& lvl: levels_) {
        if (lvl.runs.back() == lvl.pending.size()) lvl.runs.pop_back();
    }
    PL(buffer);

    // Spikes received from each domain; each domain is in at most one level.
    struct received_range {
        int domain;
        const spike* first;
        count_type count;
    };
    std::vector<received_range> ranges{{rank_, local_spikes.data(), count_type(local_spikes.size())}};
    // Keeps the received spikes alive; reserved so that the pointers in ranges stay valid.
    std::vector<gathered_vector<spike>> received;
    received.reserve(levels_.size());

    for (int l = 0; l <= far_level(); ++l) {
        auto& lvl = levels_[l];
        bool due = lvl.period && (!epoch_id || *epoch_id % std::ptrdiff_t(lvl.period) == 0);
        if (!due) {
            arb_assert(lvl.period || lvl.pending.empty());
            continue;
        }

        PE(merge);
        merge_sorted_runs(lvl.pending, lvl.runs);
        PL(merge);
        if (lvl.neighbors) {
            PE(neighborhood);
            const auto& spikes = received.emplace_back(lvl.neighbors->gather_spikes(lvl.pending));
            PL(neighborhood);
            const auto& sources = lvl.neighbors->sources();
            for (std::size_t i = 0; i < sources.size(); ++i) {
                ranges.push_back({sources[i], spikes.values().data() + spikes.partition()[i], spikes.count(i)});
            }
        }
        else {
            PE(far);
            const auto& spikes = received.emplace_back(dist_->all_to_all_spikes(bin_by_far_rank(lvl.pending)));
            PL(far);
            for (int d = 0; d < num_domains_; ++d) {
                if (recv_level_[d] != l) continue;
                ranges.push_back({d, spikes.values().data() + spikes.partition()[d], spikes.count(d)});
            }
        }
        lvl.clear();
    }

    PE(collect);
    std::vector<count_type> partition(num_domains_ + 1, 0);
    for (const auto& r: ranges) partition[r.domain + 1] = r.count;
    std::partial_sum(partition.begin(), partition.end(), partition.begin());
    // Holds unless a domain was received from twice.
    arb_assert(partition.back() == util::sum_by(ranges, [](const auto& r) { return r.count; }, count_type(0)));
    std::vector<spike> spikes(partition.back());
    for (const auto& r: ranges) std::copy_n(r.first, r.count, spikes.begin() + partition[r.domain]);
    PL(collect);
    return {std::move(spikes), std::move(partition)};
}

gathered_vector<spike>
neighborhood_router::bin_by_far_rank(const std::vector<spike>& spikes) const {
    // Counting sort, which keeps the spikes in each bin sorted by source.
    using count_type = gathered_vector<spike>::count_type;
    auto for_each_bin = [&](const spike& spk, auto&& f) {
        for (auto d: routes_.at(spk.source).ranks) {
            if (send_level_[d] == far_level()) f(d);
        }
    };
    std::vector<count_type> offsets(num_domains_ + 1, 0);
    for (const auto& spk: spikes) for_each_bin(spk, [&](auto bin) { ++offsets[bin + 1]; });
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    std::vector<spike> binned(offsets.back());
    auto next = offsets;
    for (const auto& spk: spikes) for_each_bin(spk, [&](auto bin) { binned[next[bin]++] = spk; });
    return {std::move(binned), std::move(offsets)};
}

std::vector<spike_exchange_level> neighborhood_router::levels() const {
    std::vector<spike_exchange_level> result;
    for (int l = 0; l < far_level(); ++l) {
        auto ranks = levels_[l].neighbors->sources();
        std::sort(ranks.begin(), ranks.end());
        result.push_back({std::move(ranks), levels_[l].period, false});
    }
    spike_exchange_level far{{}, levels_.back().period, true};
    for (int d = 0; d < num_domains_; ++d) {
        if (recv_level_[d] == far_level()) far.ranks.push_back(d);
    }
    result.push_back(std::move(far));
    return result;
}

void neighborhood_router::clear() {
    for (auto& lvl: levels_) lvl.clear();
}

} // namespace arb
