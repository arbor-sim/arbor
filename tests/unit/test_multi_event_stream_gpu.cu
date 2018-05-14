#include <backends/event.hpp>
#include <backends/multi_event_stream_state.hpp>

using namespace arb;

using stream_state = multi_event_stream_state<deliverable_event_data>;

namespace kernel {
__global__
void copy_marked_events_kernel(
    unsigned ci,
    stream_state state,
    deliverable_event_data* store,
    unsigned& count,
    unsigned max_ev)
{
    // use only one thread here
    if (threadIdx.x || blockIdx.x) return;

    unsigned k = 0;
    auto begin = state.ev_data+state.begin_offset[ci];
    auto end = state.ev_data+state.end_offset[ci];
    for (auto p = begin; p<end; ++p) {
        if (k>=max_ev) break;
        store[k++] = *p;
    }
    count = k;
}
}

void run_copy_marked_events_kernel(
    unsigned ci,
    stream_state state,
    deliverable_event_data* store,
    unsigned& count,
    unsigned max_ev)
{
    kernel::copy_marked_events_kernel<<<1,1>>>(ci, state, store, count, max_ev);
}

