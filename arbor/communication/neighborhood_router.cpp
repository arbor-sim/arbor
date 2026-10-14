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
    rank_(dist_->id()),
    num_domains_(dist_->size())
{
    const int num_neighborhoods = options.neighborhoods.size();
    check_on_all_ranks(*dist_,
                       dist_->min(num_neighborhoods) == dist_->max(num_neighborhoods),
                       "all ranks must pass the same number of neighborhoods.");
    bool ranks_valid = true;
    for (const auto& neighborhood: options.neighborhoods) {
        for (auto r: neighborhood) ranks_valid &= r >= 0 && r < num_domains_;
    }
    check_on_all_ranks(*dist_, ranks_valid, "rank out of range.");

    // Assign every other rank to the first neighbourhood that lists it, or the far level.
    const int far = num_neighborhoods;
    recv_level_.assign(num_domains_, far);
    recv_level_[rank_] = self_level;
    for (int l = 0; l < num_neighborhoods; ++l) {
        std::vector<int> sources;
        for (auto r: options.neighborhoods[l]) {
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

void neighborhood_router::update(const std::vector<time_type>& min_delays,
                                 const source_ranks_map& src_ranks,
                                 time_type interval) {
    arb_assert(min_delays.size() == std::size_t(num_domains_));
    clear();

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
            if (l == far_level()) r.far_ranks.push_back(d);
            if (std::find(r.levels.begin(), r.levels.end(), l) == r.levels.end()) r.levels.push_back(l);
        }
        if (!r.levels.empty()) routes_.emplace(src, std::move(r));
    }
}

gathered_vector<spike>
neighborhood_router::exchange(const std::vector<spike>& local_spikes, std::optional<std::ptrdiff_t> epoch_id) {
    using count_type = gathered_vector<spike>::count_type;

    PE(buffer);
    for (const auto& spk: local_spikes) {
        auto it = routes_.find(spk.source);
        if (it == routes_.end()) continue;
        for (auto l: it->second.levels) levels_[l].pending.push_back(spk);
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

        util::sort_by(lvl.pending, [](const spike& s) { return s.source; });
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
            const auto& spikes = received.emplace_back(exchange_far(lvl.pending));
            PL(far);
            for (int d = 0; d < num_domains_; ++d) {
                if (recv_level_[d] != l) continue;
                ranges.push_back({d, spikes.values().data() + spikes.partition()[d], spikes.count(d)});
            }
        }
        lvl.pending.clear();
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

gathered_vector<spike> neighborhood_router::exchange_far(const std::vector<spike>& spikes) const {
    // Sort spikes into bins by destination.
    using count_type = gathered_vector<spike>::count_type;
    std::vector<count_type> offsets(num_domains_ + 1, 0);
    for (const auto& spk: spikes) {
        for (auto d: routes_.at(spk.source).far_ranks) ++offsets[d + 1];
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    std::vector<spike> binned(offsets.back());
    auto next = offsets;
    for (const auto& spk: spikes) {
        for (auto d: routes_.at(spk.source).far_ranks) binned[next[d]++] = spk;
    }
    return dist_->all_to_all_spikes({std::move(binned), std::move(offsets)});
}

std::vector<std::size_t> neighborhood_router::periods() const {
    std::vector<std::size_t> result;
    for (const auto& lvl: levels_) result.push_back(lvl.period);
    return result;
}

void neighborhood_router::clear() {
    for (auto& lvl: levels_) lvl.pending.clear();
}

} // namespace arb
