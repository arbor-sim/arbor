#include <vector>
#include <gtest/gtest.h>

#include "timestep_range.hpp"
#include "backends/event.hpp"
#include "backends/gpu/event_stream.hpp"
#include "memory/memory.hpp"
#include "memory/gpu_wrappers.hpp"
#include "util/rangeutil.hpp"

using namespace arb;

namespace {
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

TEST(event_stream_gpu, mark) {
    using event_stream = gpu::event_stream<deliverable_event>;

    auto thread_pool = std::make_shared<arb::threading::task_system>();

    event_stream m(thread_pool);

    arb_deliverable_event_stream s;
    arb_deliverable_event_data d;

    timestep_range dts{0, 6, 1.0};
    EXPECT_EQ(dts.size(), 6u);

    std::vector<std::vector<deliverable_event>> events(dts.size());
    for (const auto& ev : common_events) {
        events[dts.find(event_time(ev))-dts.begin()].push_back(ev);
    }
    arb_assert(util::sum_by(events, [] (const auto& v) {return v.size();}) == common_events.size());

    m.init(events);

    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.end - s.begin, 0u);

    m.mark();
    // current time is 0: no events
    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.end - s.begin, 0u);

    m.mark();
    // current time is 1: no events
    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.end - s.begin, 0u);

    m.mark();
    // current time is 2: 1 event at mech_index 1
    EXPECT_FALSE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.end - s.begin, 1u);
    cpy_d2h(&d, s.begin+0);
    EXPECT_TRUE(event_matches(d, 0u));

    m.mark();
    // current time is 3: 2 events at mech_index 0 and 2
    EXPECT_FALSE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.end - s.begin, 2u);
    // the order of these 2 events is inverted on GPU due to sorting
    cpy_d2h(&d, s.begin+0);
    EXPECT_TRUE(event_matches(d, 2u));
    cpy_d2h(&d, s.begin+1);
    EXPECT_TRUE(event_matches(d, 1u));

    m.mark();
    // current time is 4: no events
    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.end - s.begin, 0u);

    m.mark();
    // current time is 5: 2 events at mech_index 4
    EXPECT_FALSE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.end - s.begin, 2u);
    cpy_d2h(&d, s.begin+0);
    EXPECT_TRUE(event_matches(d, 3u));
    cpy_d2h(&d, s.begin+1);
    EXPECT_TRUE(event_matches(d, 4u));

    m.mark();
    // current time is past time range
    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.end - s.begin, 0u);

    m.clear();
    // no events after clear
    EXPECT_TRUE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.end - s.begin, 0u);
}
