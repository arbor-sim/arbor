#include <set>
#include <vector>
#include <numeric>
#include <queue>

#include <arbor/common_types.hpp>

#include "io/trace.hpp"
#include "merge_events.hpp"
#include "util/tourney_tree.hpp"

namespace arb {

// k-way linear merge:
// Pick stream with the minimum element, pop that and push into output.
// Repeat.
void ARB_ARBOR_API linear_merge_events(std::vector<event_span>& sources, pse_vector& out) {
    // Consume all events.
    for (;;) {
        // Now find the minimum
        auto mevt =  spike_event{0, terminal_time, 0};;
        auto midx = -1;
        for (auto idx = 0ull; idx < sources.size(); ++idx) {
            auto& source = sources[idx];
            if (!source.empty()) {
                auto& evt = source.front();
                if (evt < mevt) {
                    mevt = evt;
                    midx = idx;
                }
            }
        }
        if (midx == -1) break;
        // Take event: bump chosen stream and stuff event into output.
        sources[midx].left++;
        out.emplace_back(mevt);
    }
}

// priority-queue based merge.
void ARB_ARBOR_API pqueue_merge_events(std::vector<event_span>& sources, pse_vector& out) {
    // Min heap tracking the minimum element from each span
    using kv_type = std::pair<spike_event, int>;
    std::priority_queue<kv_type, std::vector<kv_type>, std::greater<>> heap;

    // Add the first element from each sorted vector to the min heap
    for (std::size_t ix = 0; ix < sources.size(); ++ix) {
        auto& source = sources[ix];
        if (!source.empty()) {
            heap.emplace(source.front(), ix);
            source.left++;
        }
    }

    // Merge by continually popping the minimum element from the min heap
    while (!heap.empty()) {
        auto [value, ix] = heap.top();
        heap.pop();
        out.emplace_back(value);

        // If the sorted vector from which the minimum element was taken still
        // has elements, add the next smallest element to the heap
        auto& source = sources[ix];
        if (!source.empty()) {
            heap.emplace(source.front(), ix);
            source.left++;
        }
    }
}

void ARB_ARBOR_API merge_events(std::vector<event_span>& sources, pse_vector &out) {
    // Count events, bail if none; else allocate enough space to store them.
    auto n_evts = std::accumulate(sources.begin(), sources.end(),
                                  0,
                                  [] (auto acc, const auto& rng) { return acc + rng.size(); });
    out.reserve(out.size() + n_evts);
    auto n_queues = sources.size();
    if (n_queues < 20) { // NOTE: MAGIC NUMBER, found by ubench/merge
        linear_merge_events(sources, out);
    }
    else {
        pqueue_merge_events(sources, out);
    }
}

} // namespace arb
