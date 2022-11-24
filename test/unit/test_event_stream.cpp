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
    constexpr cell_local_size_type mech_1 = 10u;
    constexpr cell_local_size_type mech_2 = 13u;

    target_handle handle[4] = {
        target_handle(mech_1, 0u),
        target_handle(mech_2, 1u),
        target_handle(mech_1, 4u),
        target_handle(mech_2, 2u)
    };

    std::vector<deliverable_event> common_events = {
        deliverable_event(2.f, handle[1], 2.f),
        deliverable_event(3.f, handle[0], 1.f),
        deliverable_event(3.f, handle[3], 4.f),
        deliverable_event(5.f, handle[2], 3.f)
    };

    template <typename E>
    bool event_matches_handle(const E& e, unsigned handle_id) {
        return e.mech_id==handle[handle_id].mech_id
               && e.mech_index==handle[handle_id].mech_index;
    }
}

TEST(event_stream, init) {
    using event_stream = multicore::event_stream<deliverable_event>;

    event_stream m;

    auto events = common_events;
    ASSERT_TRUE(util::is_sorted_by(events, evtime));

    m.init(events);
    EXPECT_FALSE(m.empty());

    m.clear();
    EXPECT_TRUE(m.empty());
}

TEST(event_stream, mark) {
    using event_stream = multicore::event_stream<deliverable_event>;

    event_stream m;

    auto events = common_events;
    ASSERT_TRUE(util::is_sorted_by(events,
                [](deliverable_event e) { return event_time(e); }));
    m.init(events);
    // take a pointer to the first event in the sequence
    const auto B = m.marked_events().begin_marked;


    // Expect no marked events initially
    {
        auto marked = m.marked_events();
        EXPECT_TRUE(marked.begin_marked==marked.end_marked);
    }

    arb::time_type t_until = 2.5f;
    m.mark_until_after(t_until);
    {
        auto marked = m.marked_events();
        EXPECT_EQ(marked.size(), 1u);
        EXPECT_EQ(marked.begin_marked, B);
        auto& e = *marked.begin_marked;
        EXPECT_TRUE(event_matches_handle(e, 1));
        EXPECT_EQ(e.weight, 2.f);
    }
    m.drop_marked_events();

    m.mark_until_after(2.75);
    EXPECT_EQ(m.marked_events().size(), 0u);

    m.mark_until_after(4.0);
    {
        auto marked = m.marked_events();
        EXPECT_EQ(marked.size(), 2u);
        EXPECT_EQ(marked.begin_marked, B+1);
        auto e = marked.begin_marked;
        EXPECT_TRUE(event_matches_handle(e[0], 0));
        EXPECT_EQ(e[0].weight, 1.f);
        EXPECT_TRUE(event_matches_handle(e[1], 3));
        EXPECT_EQ(e[1].weight, 4.f);
    }

    m.drop_marked_events();
    EXPECT_EQ(m.marked_events().size(), 0u);

    m.mark_until_after(5.0);
    {
        auto marked = m.marked_events();
        EXPECT_EQ(marked.size(), 1u);
        EXPECT_EQ(marked.begin_marked, B+3);
        auto e = marked.begin_marked;
        EXPECT_TRUE(event_matches_handle(*e, 2));
        EXPECT_EQ(e->weight, 3.f);
    }

    m.drop_marked_events();
    EXPECT_EQ(m.marked_events().begin_marked, m.marked_events().end_marked);
}
