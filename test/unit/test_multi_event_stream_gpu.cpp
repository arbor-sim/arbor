#include <cstdio>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "memory/memory.hpp"

#include <backends/event.hpp>
#include <backends/gpu/multi_event_stream.hpp>
#include <memory/gpu_wrappers.hpp>
#include <util/rangeutil.hpp>

using namespace arb;

namespace {
    auto evtime = [](deliverable_event e) { return event_time(e); };

    constexpr cell_local_size_type mech = 13u;

    target_handle handle[4] = {
        target_handle(mech, 0u),
        target_handle(mech, 1u),
        target_handle(mech, 4u),
        target_handle(mech, 2u)
    };

    std::vector<deliverable_event> common_events = {
        deliverable_event(3.f, handle[0], 1.f),
        deliverable_event(2.f, handle[1], 2.f),
        deliverable_event(5.f, handle[1], 6.f),
        deliverable_event(3.f, handle[3], 4.f),
        deliverable_event(5.f, handle[2], 3.f)
    };

    bool event_matches(const arb_deliverable_event_data& e, unsigned i) {
        const auto& expected = common_events[i];
        return (e.weight == expected.weight &&
            e.mech_index == expected.handle.mech_index);
    }
}

using deliverable_event_stream = gpu::multi_event_stream<deliverable_event>;

TEST(multi_event_stream_gpu, init) {

    deliverable_event_stream m;

    ASSERT_TRUE(std::is_sorted(common_events.begin(), common_events.end(),
        [](const auto& a, const auto& b) {
            return a.handle.mech_index < b.handle.mech_index ||
                (a.handle.mech_index == b.handle.mech_index && event_time(a) < event_time(b));
        }));

    m.init(common_events);
    EXPECT_FALSE(m.empty());

    auto marked = m.marked_events();

    std::vector<arb_deliverable_event_data> host_data(common_events.size());
    memory::gpu_memcpy_d2h(
        host_data.data(),
        marked.ev_data,
        common_events.size()*sizeof(arb_deliverable_event_data));

    for (unsigned i=0; i<host_data.size(); ++i) {
        EXPECT_TRUE(event_matches(host_data[i], i));
    }

    m.clear();
    EXPECT_TRUE(m.empty());
}
