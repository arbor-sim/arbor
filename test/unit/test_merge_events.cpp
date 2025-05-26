#include <gtest/gtest.h>

#include <vector>
#include <ranges>

#include <arbor/event_generator.hpp>
#include <arbor/spike_event.hpp>

#include "merge_events.hpp"
#include "util/rangeutil.hpp"
#include "util/tourney_tree.hpp"

namespace arb {
void merge_cell_events(
    time_type t_from,
    time_type t_to,
    event_span old_events,
    event_span pending,
    std::vector<event_generator>& generators,
    pse_vector& new_events);
} // namespace arb

using namespace arb;

// Wrapper for arb::merge_cell_events.
static void merge_events(
    time_type t_from,
    time_type t_to,
    const pse_vector& old_events,
    pse_vector& pending,
    std::vector<event_generator>& generators,
    pse_vector& new_events) {
    std::ranges::sort(pending);
    merge_cell_events(t_from, t_to,
                      util::range_pointer_view(old_events),
                      util::range_pointer_view(pending),
                      generators,
                      new_events);
}

std::vector<event_generator> empty_gens;

// Test the trivial case of merging empty sets
TEST(merge_events, empty)
{
    pse_vector events;
    pse_vector lc;
    pse_vector lf;

    merge_events(0, terminal_time, lc, events, empty_gens, lf);

    EXPECT_EQ(lf.size(), 0u);
}


// Test the case where there are no events in lc that are to be delivered
// after tfinal.
TEST(merge_events, no_overlap)
{
    pse_vector lc = {
        {0, 1, 1},
        {0, 2, 1},
        {0, 3, 3},
    };
    // Check that the inputs satisfy the precondition that lc is sorted.
    EXPECT_TRUE(std::ranges::is_sorted(lc));

    // These events should be removed from lf by merge_events, and replaced
    // with events to be delivered after t=10
    pse_vector lf = {
        {0, 1, 1},
        {0, 2, 1},
        {0, 3, 3},
    };

    pse_vector events = {
        {0, 12, 1},
        {0, 11, 2},
        {0, 10, 4},
        {0, 11, 1},
    };

    merge_events(10, terminal_time, lc, events, empty_gens, lf);

    pse_vector expected = {
        {0, 10, 4},
        {0, 11, 1},
        {0, 11, 2},
        {0, 12, 1},
    };

    EXPECT_TRUE(std::ranges::is_sorted(lf));
    EXPECT_EQ(expected, lf);
}

// Test case where current events (lc) contains events that must be deilvered
// in a future epoch, i.e. events with delivery time greater than the tfinal
// argument passed to merge_events.
TEST(merge_events, overlap)
{
    pse_vector lc = {
        {0, 1, 1},
        {0, 2, 1},
        // The current epoch ends at t=10, so all events from here down are expected in lf.
        {0, 10, 2},
        {0, 11, 3},
    };
    EXPECT_TRUE(std::ranges::is_sorted(lc));

    pse_vector lf;

    pse_vector events = {
        // events are in reverse order: they should be sorted in the output of merge_events.
        {0, 12, 1},
        {0, 11, 2},
        {0, 11, 1},
        {1, 10, 3},
        {0, 10, 8},
    };

    merge_events(10, terminal_time, lc, events, empty_gens, lf);

    pse_vector expected = {
        {0, 10, 2}, // from lc
        {0, 10, 8}, // from events
        {1, 10, 3}, // from events
        {0, 11, 1}, // from events
        {0, 11, 2}, // from events
        {0, 11, 3}, // from lc
        {0, 12, 1}, // from events
    };

    EXPECT_TRUE(std::ranges::is_sorted(lf));
    EXPECT_EQ(expected, lf);
}

// Test the merge_events method with event generators.
TEST(merge_events, X)
{
    const time_type t0 = 10;
    const time_type t1 = 20;

    pse_vector lc = {
        {0, 1, 1},
        {0, 5, 1},
        // The current epoch ends at t=10, so all events from here down are expected in lf.
        {0, 10, 2},
        {0, 11, 3},
        {0, 20, 2},
        {0, 21, 3},
    };
    EXPECT_TRUE(std::ranges::is_sorted(lc));

    pse_vector lf;

    pse_vector events = {
        {0, 12, 1},
        {0, 15, 2},
        {0, 22, 3},
        {0, 26, 4},
    };

    std::vector<event_generator> generators = {regular_generator({"l0"}, 42.f, t0*arb::units::ms, 5*arb::units::ms)};
    generators[0].set_target_lid(2);
    merge_events(t0, t1, lc, events, generators, lf);

    pse_vector expected = {
        {0, 10, 2},  // from lc
        {2, 10, 42}, // from generator
        {0, 11, 3},  // from lc
        {0, 12, 1},  // from events
        {0, 15, 2},  // from events
        {2, 15, 42}, // from generator
        {0, 20, 2},  // from lc
        {0, 21, 3},  // from lc
        {0, 22, 3},  // from events
        {0, 26, 4},  // from events
    };

    EXPECT_TRUE(std::ranges::is_sorted(lf));
    EXPECT_EQ(expected, lf);
}

struct labeled_synapse_event {};

using lse_vector = std::vector<std::tuple<cell_local_label_type, time_type, float>>;

// Test the tournament tree for merging two small sequences 
TEST(merge_events, tourney_seq)
{
    std::vector<arb::time_type> times {1, 2, 3, 4, 5};
    cell_local_label_type l0 = {"l0"};
    float w1 = 1.0f, w2 = 2.0f;
    lse_vector evs1, evs2;
    pse_vector expected;
    for (const auto time: times) {
        evs1.emplace_back(l0, w1, time);
        evs2.emplace_back(l0, w2, time);
        expected.push_back({0, time, w1});
        expected.push_back({0, time, w2});
    }
    std::ranges::sort(expected);

    auto g1 = explicit_generator_from_milliseconds(l0, w1, times);
    g1.set_target_lid(0);
    auto g2 = explicit_generator_from_milliseconds(l0, w2, times);
    g2.set_target_lid(0);

    std::vector<event_span> spans = {g1.events(0, terminal_time),
                                     g2.events(0, terminal_time)};
    tourney_tree tree(spans);

    pse_vector lf;
    while (!tree.empty()) {
        lf.push_back(tree.head());
        tree.pop();
    }

    EXPECT_TRUE(std::ranges::is_sorted(lf));
    EXPECT_EQ(expected, lf);
}

// Test the tournament tree on a large set of Poisson generators.
TEST(merge_events, tourney_poisson) {
    // Number of poisson generators.
    // Not a power of 2, so that there will be "null" leaf nodes in the
    // tournament tree.
    auto ngen = 100u;
    time_type tfinal = 10;
    time_type t0 = 0;
    time_type lambda = 10; // expected: tfinal*lambda=1000 events per generator

    std::vector<event_generator> generators;
    for (auto i=0u; i<ngen; ++i) {
        cell_lid_type lid = i;
        cell_tag_type label = "tgt"+std::to_string(i);
        float weight = i;
        // the first and last generators have the same seed to test that sorting
        // of events with the same time but different weights works properly.
        auto G = i%(ngen-1);
        auto gen = poisson_generator(label, weight, t0*arb::units::ms, lambda*arb::units::kHz, G);
        gen.set_target_lid(lid);
        generators.push_back(std::move(gen));
    }

    // manually generate the expected output
    pse_vector expected;
    for (auto& gen: generators) {
        // Push all events before tfinal in gen to the expected values.
        event_span evs = gen.events(t0, tfinal);
        util::append(expected, evs);

        // Reset the generator so that it is ready to generate the same
        // events again for the tournament tree test.
        gen.reset();
    }
    // Manually sort the expected events.
    std::ranges::sort(expected);

    // Generate output using tournament tree in lf.
    std::vector<event_span> spans;
    for (auto& gen: generators) {
        spans.emplace_back(gen.events(t0, tfinal));
    }
    tourney_tree tree(spans);
    pse_vector lf;
    while (!tree.empty()) {
        lf.push_back(tree.head());
        tree.pop();
    }

    // Test output of tournament tree.
    EXPECT_TRUE(std::ranges::is_sorted(lf));
    EXPECT_EQ(lf, expected);
}
