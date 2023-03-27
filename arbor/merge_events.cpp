#include <vector>
#include <numeric>

#include <arbor/common_types.hpp>


#include "io/trace.hpp"
#include "merge_events.hpp"
#include "util/tourney_tree.hpp"

namespace arb {

// Simple alternative
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

void merge_events(std::vector<event_span> sources, pse_vector &out) {
    // Count events, bail if none; else allocate enough space to store them.
    auto n_evts = std::accumulate(sources.begin(), sources.end(),
                                  0,
                                  [] (auto acc, const auto& rng) { return acc + rng.size(); });
    if (n_evts == 0) return;
    out.reserve(out.size() + n_evts);

}

} // namespace arb
