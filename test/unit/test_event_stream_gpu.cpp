#include "backends/gpu/event_stream.hpp"
#include "memory/memory.hpp"
#include "memory/gpu_wrappers.hpp"
#include "util/rangeutil.hpp"
#include "./test_event_stream.hpp"

namespace {

template<typename T>
void cpy_d2h(T* dst, const T* src, std::size_t n) {
    memory::gpu_memcpy_d2h(dst, src, sizeof(T)*n);
}

template<typename Result>
void check(Result result) {
    for (std::size_t step=0; step<result.steps.size(); ++step) {
        for (auto& [mech_id, stream] :  result.streams) {
            stream.mark();
            auto marked = stream.marked_events();
            std::vector<arb_deliverable_event_data> host_data(marked.end - marked.begin);
            EXPECT_EQ(host_data.size(), result.expected[mech_id][step].size());
            if (host_data.size()) {
                cpy_d2h(host_data.data(), marked.begin, host_data.size());
                check_result(host_data.data(), result.expected[mech_id][step]);
            }
        }
    }
}

}

TEST(event_stream_gpu, single_step) {
    check(single_step<gpu::spike_event_stream>());
}

TEST(event_stream_gpu, multi_step) {
    check(multi_step<gpu::spike_event_stream>());
}
