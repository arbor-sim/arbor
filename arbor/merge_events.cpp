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
void linear_merge_events(std::vector<event_span> sources, pse_vector& out) {
    // Consume all events.
    for (;;) {
        // Discard empty streams and bail if none remain.
        sources.erase(std::remove_if(sources.begin(),
                                     sources.end(),
                                     [](auto rng){ return rng.empty(); }),
                      sources.end());
        if (sources.empty()) break;
        // Now find the minimum
        auto mevt =  spike_event{0, terminal_time, 0};;
        auto midx = -1;
        for (auto idx = 0ull; idx < sources.size(); ++idx) {
            // SAFETY: There are no empty streams, since we ditched those above.
            auto& evt = sources[idx].front();
            if (evt < mevt) {
                mevt = evt;
                midx = idx;
            }
        }
        // Take event: bump chosen stream and stuff event into output.
        sources[midx].left++;
        out.emplace_back(mevt);
    }
}

// priority-queue based merge.
void pqueue_merge_events(std::vector<event_span> sources, pse_vector& out) {
    // Create a priority queue to keep track of the minimum element from each sorted vector
    std::priority_queue<std::pair<spike_event, int>,
                        std::vector<std::pair<spike_event, int>>,
                        std::greater<>> heap;

    // Add the first element from each sorted vector to the min heap
    for (std::size_t ix = 0; ix < sources.size(); ++ix) {
        auto& source = sources[ix];
        heap.emplace(source.front(), ix);
        source.left++;
    }

    // Merge the sorted vectors by continually popping the minimum element from the min heap
    while (!heap.empty()) {
        auto [value, idx] = heap.top();
        heap.pop();
        out.emplace_back(value);

        // If the sorted vector from which the minimum element was taken still has elements,
        // add the next element to the min heap
        auto& source = sources[idx];
        if (!source.empty()) {
            heap.emplace(source.front(), idx);
            source.left++;
        }
    }
}

void merge_events(std::vector<event_span> sources, pse_vector &out) {
    // Count events, bail if none; else allocate enough space to store them.
    auto n_evts = std::accumulate(sources.begin(), sources.end(),
                                  0,
                                  [] (auto acc, const auto& rng) { return acc + rng.size(); });
    if (n_evts == 0) return;
    out.reserve(out.size() + n_evts);
    auto n_queues = sources.size();
    if (n_queues < 20) { // MAGIC: Found by ubench/merge
        linear_merge_events(std::move(sources), out);
    }
    else {
        pqueue_merge_events(std::move(sources), out);
    }
}

} // namespace arb
