#include "../gtest.h"

#include <event_queue.hpp>
#include <model.hpp>

namespace arb {
    // Declare prototype of the merge_events function, because it is only
    // defined in the TU of model.cpp
    void merge_events(
        event_vector& events,
        const event_vector& lc,
        event_vector& lf,
        time_type tfinal);
}

using namespace arb;

std::ostream& operator<<(std::ostream& o, const event_vector& events) {
    o << "{{"; for (auto e: events) o << " " << e;
    o << "}}";
    return o;
}

using pse = postsynaptic_spike_event;
auto ev_bind = [] (const pse& e){ return std::tie(e.time, e.target, e.weight); };
auto ev_less = [] (const pse& l, const pse& r){ return ev_bind(l)<ev_bind(r); };

// Test the trivial case of merging empty sets
TEST(merge_events, empty)
{
    event_vector events;
    event_vector lc;
    event_vector lf;

    merge_events(events, lc, lf, 0);

    EXPECT_EQ(lf.size(), 0u);
}

// Test the case where there are no events in lc that are to be delivered
// after tfinal.
TEST(merge_events, no_overlap)
{
    event_vector lc = {
        {{0, 0}, 1, 1},
        {{0, 0}, 2, 1},
        {{0, 0}, 3, 3},
    };
    // Check that the inputs satisfy the precondition that lc is sorted.
    EXPECT_TRUE(std::is_sorted(lc.begin(), lc.end(), ev_less));

    // These events should be removed from lf by merge_events, and replaced
    // with events to be delivered after t=10
    event_vector lf = {
        {{0, 0}, 1, 1},
        {{0, 0}, 2, 1},
        {{0, 0}, 3, 3},
    };

    event_vector events = {
        {{0, 0}, 12, 1},
        {{0, 0}, 11, 2},
        {{8, 0}, 10, 4},
        {{0, 0}, 11, 1},
    };

    merge_events(events, lc, lf, 10);

    event_vector expected = {
        {{8, 0}, 10, 4},
        {{0, 0}, 11, 1},
        {{0, 0}, 11, 2},
        {{0, 0}, 12, 1},
    };

    EXPECT_TRUE(std::is_sorted(lf.begin(), lf.end(), ev_less));
    EXPECT_EQ(expected, lf);
}

// Test case where current events (lc) contains events that must be deilvered
// in a future epoch, i.e. events with delivery time greater than the tfinal
// argument passed to merge_events.
TEST(merge_events, overlap)
{
    event_vector lc = {
        {{0, 0}, 1, 1},
        {{0, 0}, 2, 1},
        // The current epoch ends at t=10, so all events from here down are expected in lf.
        {{8, 0}, 10, 2},
        {{0, 0}, 11, 3},
    };
    EXPECT_TRUE(std::is_sorted(lc.begin(), lc.end(), ev_less));

    event_vector lf;

    event_vector events = {
        // events are in reverse order: they should be sorted in the output of merge_events.
        {{0, 0}, 12, 1},
        {{0, 0}, 11, 2},
        {{0, 0}, 11, 1},
        {{8, 0}, 10, 3},
        {{7, 0}, 10, 8},
    };

    merge_events(events, lc, lf, 10);

    event_vector expected = {
        {{7, 0}, 10, 8}, // from events
        {{8, 0}, 10, 2}, // from lc
        {{8, 0}, 10, 3}, // from events
        {{0, 0}, 11, 1}, // from events
        {{0, 0}, 11, 2}, // from events
        {{0, 0}, 11, 3}, // from lc
        {{0, 0}, 12, 1}, // from events
    };

    EXPECT_TRUE(std::is_sorted(lf.begin(), lf.end(), ev_less));
    EXPECT_EQ(expected, lf);
}
