#include <vector>
#include <gtest/gtest.h>

#include "backends/event.hpp"
#include "backends/multicore/multi_event_stream.hpp"
#include "util/rangeutil.hpp"

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
}

TEST(multi_event_stream, mark) {
    using multi_event_stream = multicore::multi_event_stream;

    multi_event_stream m;

    ASSERT_TRUE(std::is_sorted(common_events.begin(), common_events.end(),
        [](const auto& a, const auto& b) { return event_time(a) < event_time(b);}));

    arb_deliverable_event_stream s;
    arb_deliverable_event_range r;
    event_map em;

    timestep_range dts{0, 6, 1.0};
    EXPECT_EQ(dts.size(), 6u);

    for (const auto& e : common_events) add_event(em, e);

    m.init(em[mech], dts);

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
    r = s.ranges[0];
    EXPECT_EQ(r.mech_index, 1u);
    EXPECT_EQ(r.end - r.begin, 1u);
    EXPECT_TRUE(event_matches(s.data[r.begin], 0u));

    m.mark();
    // current time is 3: 2 events at mech_index 0 and 2
    EXPECT_FALSE(m.empty());
    s = m.marked_events();
    EXPECT_EQ(s.num_streams, 2u);
    r = s.ranges[0];
    EXPECT_EQ(r.mech_index, 0u);
    EXPECT_EQ(r.end - r.begin, 1u);
    EXPECT_TRUE(event_matches(s.data[r.begin], 2u));
    r = s.ranges[1];
    EXPECT_EQ(r.mech_index, 2u);
    EXPECT_EQ(r.end - r.begin, 1u);
    EXPECT_TRUE(event_matches(s.data[r.begin], 1u));

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
    r = s.ranges[0];
    EXPECT_EQ(r.mech_index, 4u);
    EXPECT_EQ(r.end - r.begin, 2u);
    EXPECT_TRUE(event_matches(s.data[r.begin+0u], 3u));
    EXPECT_TRUE(event_matches(s.data[r.begin+1u], 4u));

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
