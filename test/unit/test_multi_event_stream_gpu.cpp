#include <cstdio>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <backends/event.hpp>
#include <backends/gpu/multi_event_stream.hpp>
#include <memory/memory.hpp>
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
        deliverable_event(2.0, handle[1], 2.f),
        deliverable_event(3.0, handle[3], 4.f),
        deliverable_event(3.0, handle[0], 1.f),
        deliverable_event(5.0, handle[2], 3.f),
        deliverable_event(5.5, handle[2], 6.f)
    };

    bool event_matches(const arb_deliverable_event_data& e, unsigned i) {
        const auto& expected = common_events[i];
        return (e.weight == expected.weight);
    }

    template<typename T>
    void cpy_d2h(T* dst, const T* src) {
        memory::gpu_memcpy_d2h(dst, src, sizeof(T));
    }
}

TEST(multi_event_stream_gpu, mark) {
    using multi_event_stream = gpu::multi_event_stream;

    multi_event_stream m;

    ASSERT_TRUE(std::is_sorted(common_events.begin(), common_events.end(),
        [](const auto& a, const auto& b) { return event_time(a) < event_time(b);}));

    arb_deliverable_event_stream s;
    arb_deliverable_event_range r;
    arb_deliverable_event_data d;

    timestep_range dts{0, 6, 1.0};
    EXPECT_EQ(dts.size(), 6u);

    std::vector<deliverable_event> events(common_events);
    std::stable_sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.handle.mech_index < rhs.handle.mech_index; });
    m.init(events, mech, common_events.size(), dts);

    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.num_streams, 0u);

    m.mark();
    // current time is 0: no events
    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.num_streams, 0u);

    m.mark();
    // current time is 1: no events
    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.num_streams, 0u);

    m.mark();
    // current time is 2: 1 event at mech_index 1
    EXPECT_FALSE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.num_streams, 1u);
    cpy_d2h(&r, s.ranges);
    EXPECT_EQ(r.mech_index, 1u);
    EXPECT_EQ(r.end - r.begin, 1u);
    cpy_d2h(&d, s.data + r.begin);
    EXPECT_TRUE(event_matches(d, 0u));

    m.mark();
    // current time is 3: 2 events at mech_index 0 and 2
    EXPECT_FALSE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.num_streams, 2u);
    cpy_d2h(&r, s.ranges+0);
    EXPECT_EQ(r.mech_index, 0u);
    EXPECT_EQ(r.end - r.begin, 1u);
    cpy_d2h(&d, s.data + r.begin);
    EXPECT_TRUE(event_matches(d, 2u));
    cpy_d2h(&r, s.ranges+1);
    EXPECT_EQ(r.mech_index, 2u);
    EXPECT_EQ(r.end - r.begin, 1u);
    cpy_d2h(&d, s.data + r.begin);
    EXPECT_TRUE(event_matches(d, 1u));

    m.mark();
    // current time is 4: no events
    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.num_streams, 0u);

    m.mark();
    // current time is 5: 2 events at mech_index 4
    EXPECT_FALSE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.num_streams, 1u);
    cpy_d2h(&r, s.ranges);
    EXPECT_EQ(r.mech_index, 4u);
    EXPECT_EQ(r.end - r.begin, 2u);
    cpy_d2h(&d, s.data + r.begin+0u);
    EXPECT_TRUE(event_matches(d, 3u));
    cpy_d2h(&d, s.data + r.begin+1u);
    EXPECT_TRUE(event_matches(d, 4u));

    m.mark();
    // current time is past time range
    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.num_streams, 0u);

    m.clear();
    // no events after clear
    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.num_streams, 0u);
}
