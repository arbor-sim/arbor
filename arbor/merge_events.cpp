#include <vector>
#include <numeric>

#include <arbor/common_types.hpp>

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

template <typename T>
static inline void sift_down_heap(std::vector<T>& heap, size_t len, auto&& comp=std::greater<>{}) {
    auto node = std::move(heap[0]);
    std::size_t ix = 0;
    for (;;) {
        auto cx = 2 * ix + 1;
        if (cx >= len) break;
        if (cx + 1 < len && comp(heap[cx], heap[cx + 1])) ++cx; // pick the smaller child
        if (!comp(node, heap[cx])) break;                       // node <= smaller child: done
        heap[ix] = std::move(heap[cx]);
        ix       = cx;
    }
    heap[ix] = std::move(node);
}

void ARB_ARBOR_API pqueue_merge_events(std::vector<event_span>& sources, pse_vector& out) {
    // Min heap tracking the minimum element from each span
    using kv_type = std::pair<spike_event, int>;
    std::vector<kv_type> heap;
    heap.reserve(sources.size());

    // Seed with the head of every non-empty stream, then build in O(k).
    for (std::size_t ix = 0; ix < sources.size(); ++ix) {
        auto& s = sources[ix];
        if (s.empty()) continue;
        heap.emplace_back(s.front(), ix);
        ++s.left;
    }
    std::make_heap(heap.begin(), heap.end(), std::greater<>{});

    auto beg = heap.begin();
    auto end = heap.end();

    // Merge by continually popping the minimum element from the min heap
    while (end > beg) {
        auto& [value, ix] = heap.front();
        out.emplace_back(value);
        auto& s = sources[ix];
        if (!s.empty()) {
            heap.front().first = s.front();
            ++s.left;
            sift_down_heap(heap, end - beg, std::greater<>{});
        }
        else {
            std::pop_heap(beg, end, std::greater<>{});
            --end;
        }
    }
}

void ARB_ARBOR_API merge_events(std::vector<event_span>& sources, pse_vector &out, std::size_t n_evts) {
    out.reserve(out.size() + n_evts);
    auto n_queues = sources.size();
    if (n_queues < 15) { // NOTE: MAGIC NUMBER, found by ubench/merge
        linear_merge_events(sources, out);
    }
    else {
        pqueue_merge_events(sources, out);
    }
}

void ARB_ARBOR_API merge_events(std::vector<event_span>& sources, pse_vector &out) {
    // Count events, bail if none; else allocate enough space to store them.
    auto n_evts = std::accumulate(sources.begin(), sources.end(),
                                  0,
                                  [] (auto acc, const auto& rng) { return acc + rng.size(); });
    merge_events(sources, out, n_evts);
}


} // namespace arb
