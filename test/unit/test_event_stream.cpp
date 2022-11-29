#include <vector>
#include <gtest/gtest.h>

#include "arbor/common_types.hpp"
#include "backends/event.hpp"
#include "backends/multicore/event_stream.hpp"
#include "util/rangeutil.hpp"

using namespace arb;

namespace {
    auto evtime = [](deliverable_event e) { return event_time(e); };
}

namespace {
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

TEST(event_stream, init) {
    using event_stream = multicore::event_stream<deliverable_event>;

    event_stream m;

    ASSERT_TRUE(std::is_sorted(common_events.begin(), common_events.end(),
        [](const auto& a, const auto& b) {
            return a.handle.mech_index < b.handle.mech_index ||
                (a.handle.mech_index == b.handle.mech_index && event_time(a) < event_time(b));
        }));

    m.init(common_events);
    EXPECT_FALSE(m.empty());

    m.clear();
    EXPECT_TRUE(m.empty());
}

TEST(event_stream, mark) {
    using event_stream = multicore::event_stream<deliverable_event>;

    event_stream m;

    const auto& events = common_events;
    m.init(events);

    // stream is not empty
    EXPECT_FALSE(m.empty());
    // 4 different mech indices
    EXPECT_EQ(m.kind_count(), 4u);
    // 5 events in total
    EXPECT_EQ(m.remaining_count(), 5u);
    // 0 marked events so far
    EXPECT_EQ(m.marked_count(), 0u);

    // Expect no marked events initially
    {
        auto marked = m.marked_events();
        EXPECT_EQ(marked.kinds, 4u);
        EXPECT_EQ(marked.marked, 0u);
        for (arb_size_type i=0; i<marked.kinds; ++i) {
            EXPECT_EQ(marked.begin_marked[i], marked.end_marked[i]);
        }
    }

    m.mark_until_after(2.5);
    EXPECT_FALSE(m.empty());
    EXPECT_EQ(m.remaining_count(), 5u);
    EXPECT_EQ(m.marked_count(), 1u);
    EXPECT_EQ(m.marked_events().marked, 1u);
    {
        auto marked = m.marked_events();
        const auto B = marked.data;
        EXPECT_EQ(marked.marked, 1u);
        EXPECT_EQ(marked.end_marked[1] - marked.begin_marked[1], 1u);
        EXPECT_EQ(marked.data + marked.begin_marked[1], B + 1);
        EXPECT_TRUE(event_matches(*(B + 1), 1));
    }

    m.drop_marked_events();
    EXPECT_FALSE(m.empty());
    EXPECT_EQ(m.remaining_count(), 4u);
    EXPECT_EQ(m.marked_count(), 0u);
    EXPECT_EQ(m.marked_events().marked, 0u);

    m.mark_until_after(2.75);
    EXPECT_FALSE(m.empty());
    EXPECT_EQ(m.remaining_count(), 4u);
    EXPECT_EQ(m.marked_count(), 0u);
    EXPECT_EQ(m.marked_events().marked, 0u);

    m.mark_until_after(4.0);
    EXPECT_FALSE(m.empty());
    EXPECT_EQ(m.remaining_count(), 4u);
    EXPECT_EQ(m.marked_count(), 2u);
    EXPECT_EQ(m.marked_events().marked, 2u);
    {
        auto marked = m.marked_events();
        const auto B = marked.data;
        EXPECT_EQ(marked.marked, 2u);
        EXPECT_EQ(marked.end_marked[0] - marked.begin_marked[0], 1u);
        EXPECT_EQ(marked.data + marked.begin_marked[0], B + 0);
        EXPECT_TRUE(event_matches(*(B + 0), 0));
        EXPECT_EQ(marked.end_marked[2] - marked.begin_marked[2], 1u);
        EXPECT_EQ(marked.data + marked.begin_marked[2], B + 3);
        EXPECT_TRUE(event_matches(*(B + 3), 3));
    }

    m.drop_marked_events();
    EXPECT_FALSE(m.empty());
    EXPECT_EQ(m.remaining_count(), 2u);
    EXPECT_EQ(m.marked_count(), 0u);
    EXPECT_EQ(m.marked_events().marked, 0u);

    m.mark_until_after(5.0);
    EXPECT_FALSE(m.empty());
    EXPECT_EQ(m.remaining_count(), 2u);
    EXPECT_EQ(m.marked_count(), 2u);
    EXPECT_EQ(m.marked_events().marked, 2u);
    {
        auto marked = m.marked_events();
        const auto B = marked.data;
        EXPECT_EQ(marked.marked, 2u);
        EXPECT_EQ(marked.end_marked[1] - marked.begin_marked[1], 1u);
        EXPECT_EQ(marked.data + marked.begin_marked[1], B + 2);
        EXPECT_TRUE(event_matches(*(B + 2), 2));
        EXPECT_EQ(marked.end_marked[3] - marked.begin_marked[3], 1u);
        EXPECT_EQ(marked.data + marked.begin_marked[3], B + 4);
        EXPECT_TRUE(event_matches(*(B + 4), 4));
    }

    m.drop_marked_events();
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.remaining_count(), 0u);
    EXPECT_EQ(m.marked_count(), 0u);
    EXPECT_EQ(m.marked_events().marked, 0u);
}
