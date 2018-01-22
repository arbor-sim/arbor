#include "../gtest.h"

#include <event_queue.hpp>
#include <merge_events.hpp>
#include <model.hpp>

using namespace arb;

std::vector<event_generator_ptr> empty_gens;

// Test the trivial case of merging empty sets
TEST(merge_events, empty)
{
    pse_vector events;
    pse_vector lc;
    pse_vector lf;

    merge_events(0, max_time, lc, events, empty_gens, lf);

    EXPECT_EQ(lf.size(), 0u);
}

// Test the case where there are no events in lc that are to be delivered
// after tfinal.
TEST(merge_events, no_overlap)
{
    pse_vector lc = {
        {{0, 0}, 1, 1},
        {{0, 0}, 2, 1},
        {{0, 0}, 3, 3},
    };
    // Check that the inputs satisfy the precondition that lc is sorted.
    EXPECT_TRUE(std::is_sorted(lc.begin(), lc.end()));

    // These events should be removed from lf by merge_events, and replaced
    // with events to be delivered after t=10
    pse_vector lf = {
        {{0, 0}, 1, 1},
        {{0, 0}, 2, 1},
        {{0, 0}, 3, 3},
    };

    pse_vector events = {
        {{0, 0}, 12, 1},
        {{0, 0}, 11, 2},
        {{8, 0}, 10, 4},
        {{0, 0}, 11, 1},
    };

    merge_events(10, max_time, lc, events, empty_gens, lf);

    pse_vector expected = {
        {{8, 0}, 10, 4},
        {{0, 0}, 11, 1},
        {{0, 0}, 11, 2},
        {{0, 0}, 12, 1},
    };

    EXPECT_TRUE(std::is_sorted(lf.begin(), lf.end()));
    EXPECT_EQ(expected, lf);
}

// Test case where current events (lc) contains events that must be deilvered
// in a future epoch, i.e. events with delivery time greater than the tfinal
// argument passed to merge_events.
TEST(merge_events, overlap)
{
    pse_vector lc = {
        {{0, 0}, 1, 1},
        {{0, 0}, 2, 1},
        // The current epoch ends at t=10, so all events from here down are expected in lf.
        {{8, 0}, 10, 2},
        {{0, 0}, 11, 3},
    };
    EXPECT_TRUE(std::is_sorted(lc.begin(), lc.end()));

    pse_vector lf;

    pse_vector events = {
        // events are in reverse order: they should be sorted in the output of merge_events.
        {{0, 0}, 12, 1},
        {{0, 0}, 11, 2},
        {{0, 0}, 11, 1},
        {{8, 0}, 10, 3},
        {{7, 0}, 10, 8},
    };

    merge_events(10, max_time, lc, events, empty_gens, lf);

    pse_vector expected = {
        {{7, 0}, 10, 8}, // from events
        {{8, 0}, 10, 2}, // from lc
        {{8, 0}, 10, 3}, // from events
        {{0, 0}, 11, 1}, // from events
        {{0, 0}, 11, 2}, // from events
        {{0, 0}, 11, 3}, // from lc
        {{0, 0}, 12, 1}, // from events
    };

    EXPECT_TRUE(std::is_sorted(lf.begin(), lf.end()));
    EXPECT_EQ(expected, lf);
}

// Test the merge_events method with event generators.
TEST(merge_events, X)
{
    const time_type t0 = 10;
    const time_type t1 = 20;

    pse_vector lc = {
        {{0, 0}, 1, 1},
        {{0, 0}, 5, 1},
        // The current epoch ends at t=10, so all events from here down are expected in lf.
        {{8, 0}, 10, 2},
        {{0, 0}, 11, 3},
        {{8, 0}, 20, 2},
        {{0, 0}, 21, 3},
    };
    EXPECT_TRUE(std::is_sorted(lc.begin(), lc.end()));

    pse_vector lf;

    pse_vector events = {
        {{0, 0}, 12, 1},
        {{1, 0}, 15, 2},
        {{2, 0}, 22, 3},
        {{3, 0}, 26, 4},
    };

    std::vector<event_generator_ptr> generators(2);
    generators.push_back(
        make_event_generator<regular_generator>
        (cell_member_type{4,2}, 42.f, t0, 5));

    merge_events(t0, t1, lc, events, generators, lf);

    pse_vector expected = {
        {{4, 2}, 10, 42}, // from generator
        {{8, 0}, 10, 2},  // from lc
        {{0, 0}, 11, 3},  // from lc
        {{0, 0}, 12, 1},  // from events
        {{1, 0}, 15, 2},  // from events
        {{4, 2}, 15, 42}, // from generator
        {{8, 0}, 20, 2},  // from lc
        {{0, 0}, 21, 3},  // from lc
        {{2, 0}, 22, 3},  // from events
        {{3, 0}, 26, 4},  // from events
    };

    EXPECT_TRUE(std::is_sorted(lf.begin(), lf.end()));
    EXPECT_EQ(expected, lf);
}

// Test the tournament tree for merging two small sequences 
TEST(merge_events, tourney_seq)
{
    pse_vector g1 = {
        {{0, 0}, 1, 1},
        {{0, 0}, 2, 2},
        {{0, 0}, 3, 3},
        {{0, 0}, 4, 4},
        {{0, 0}, 5, 5},
    };

    pse_vector g2 = {
        {{0, 0}, 1.5, 1},
        {{0, 0}, 2.5, 2},
        {{0, 0}, 3.5, 3},
        {{0, 0}, 4.5, 4},
        {{0, 0}, 5.5, 5},
    };

    std::vector<event_generator_ptr> generators;
    generators.push_back(make_event_generator<seq_generator<pse_vector>>(g1));
    generators.push_back(make_event_generator<seq_generator<pse_vector>>(g2));
    impl::tourney_tree tree(generators);

    pse_vector lf;
    while (!tree.empty()) {
        lf.push_back(tree.head());
        tree.pop();
    }

    EXPECT_TRUE(std::is_sorted(lf.begin(), lf.end()));
    auto expected = g1;
    util::append(expected, g2);
    util::sort(expected);

    EXPECT_EQ(expected, lf);
}

// Test the tournament tree on a large set of Poisson generators.
TEST(merge_events, tourney_poisson)
{
    using rndgen = std::mt19937_64;
    // Number of poisson generators.
    // Not a power of 2, so that there will be "null" leaf nodes in the
    // tournament tree.
    auto ngen = 100u;
    time_type tfinal = 10;
    time_type t0 = 0;
    time_type lambda = 10; // expected: tfinal*lambda=1000 events per generator

    std::vector<event_generator_ptr> generators;
    for (auto i=0u; i<ngen; ++i) {
        cell_member_type tgt{0, i};
        float weight = i;
        // the first and last generators have the same seed to test that sorting
        // of events with the same time but different weights works properly.
        rndgen G(i%(ngen-1));
        generators.push_back(
            make_event_generator<
                poisson_generator<std::mt19937_64>>
                (tgt, weight, G, t0, lambda));
    }

    // manually generate the expected output
    pse_vector expected;
    for (auto& gen: generators) {
        // Push all events before tfinal in gen to the expected values.
        while (gen->next().time<tfinal) {
            expected.push_back(gen->next());
            gen->pop();
        }
        // Reset the generator so that it is ready to generate the same
        // events again for the tournament tree test.
        gen->reset();
    }
    // Manually sort the expected events.
    util::sort(expected);

    // Generate output using tournament tree in lf.
    impl::tourney_tree tree(generators);
    pse_vector lf;
    while (!tree.empty(tfinal)) {
        lf.push_back(tree.head());
        tree.pop();
    }

    // Test output of tournament tree.
    EXPECT_TRUE(std::is_sorted(lf.begin(), lf.end()));
    EXPECT_EQ(lf, expected);
}
