#include <gtest/gtest.h>

#include <utility>

#include <arbor/event_generator.hpp>
#include <arbor/spike_event.hpp>

#include "util/rangeutil.hpp"

#include "common.hpp"

using namespace arb;

namespace{
    pse_vector as_vector(event_seq s) {
        return pse_vector(s.first, s.second);
    }
}

TEST(event_generators, assign_and_copy) {
    event_generator gen = regular_generator({"l2"}, 5., 0.5, 0.75);
    gen.resolve_label([](const cell_local_label_type&) {return 2;});
    spike_event expected{2, 0.75, 5.};

    auto first = [](const event_seq& seq) {
        if (seq.first==seq.second) throw std::runtime_error("no events");
        return *seq.first;
    };

    ASSERT_EQ(expected, first(gen.events(0., 1.)));
    gen.reset();

    event_generator g1(gen);
    EXPECT_EQ(expected, first(g1.events(0., 1.)));

    event_generator g2 = gen;
    EXPECT_EQ(expected, first(g2.events(0., 1.)));

    const auto& const_gen = gen;

    event_generator g3(const_gen);
    EXPECT_EQ(expected, first(g3.events(0., 1.)));

    event_generator g4 = gen;
    EXPECT_EQ(expected, first(g4.events(0., 1.)));

    event_generator g5(std::move(gen));
    EXPECT_EQ(expected, first(g5.events(0., 1.)));
}

TEST(event_generators, regular) {
    // Make a regular generator that generates its first event at t=2ms and subsequent
    // events regularly spaced 0.5 ms apart.
    time_type t0 = 2.0;
    time_type dt = 0.5;
    cell_tag_type label = "label";
    cell_lid_type lid = 3;
    float weight = 3.14;

    event_generator gen = regular_generator(label, weight, t0, dt);
    gen.resolve_label([lid](const cell_local_label_type&) {return lid;});

    // Helper for building a set of expected events.
    auto expected = [&] (std::vector<time_type> times) {
        pse_vector events;
        for (auto t: times) {
            events.push_back({lid, t, weight});
        }
        return events;
    };

    EXPECT_EQ(expected({2.0, 2.5, 3.0, 3.5, 4.0, 4.5}), as_vector(gen.events(0, 5)));

    // Test reset and re-generate.
    gen.reset();
    EXPECT_EQ(expected({2.0, 2.5, 3.0, 3.5, 4.0, 4.5}), as_vector(gen.events(0, 5)));

    // Test later intervals.
    EXPECT_EQ(expected({10.5}), as_vector(gen.events(10.1, 10.7)));
    EXPECT_EQ(expected({12, 12.5}), as_vector(gen.events(12, 12.7)));
}

using lse_vector = std::vector<std::tuple<cell_local_label_type, time_type, float>>;

TEST(event_generators, seq) {
    std::vector<arb::time_type> times = {1, 2, 3, 4, 5, 6, 7};
    lse_vector in;
    pse_vector expected;
    float weight = 0.42;
    arb::cell_local_label_type l0 = {"l0"};
    for (auto time: times) {
        in.push_back({l0, weight, time});
        expected.push_back({0, time, weight});
    }

    event_generator gen = explicit_generator(l0, weight, times);
    gen.resolve_label([](const cell_local_label_type&) {return 0;});

    EXPECT_EQ(expected, as_vector(gen.events(0, 100.))); gen.reset();
    EXPECT_EQ(expected, as_vector(gen.events(0, 100.))); gen.reset();

    auto draw = [](auto& gen, auto t0, auto t1) { gen.reset(); return as_vector(gen.events(t0, t1)); };
    auto events = [&expected] (int b, int e) { auto beg = expected.begin(); return pse_vector(beg+b, beg+e); };

    // a range that includes all the events
    EXPECT_EQ(expected, draw(gen, 0, 8));

    // a strict subset including the first event
    EXPECT_EQ(events(0, 2), draw(gen, 0, 3));

    // a strict subset including the last event
    EXPECT_EQ(events(2, 4), draw(gen, 3.0, 5));

    // subset that excludes first and last entries
    EXPECT_EQ(events(1, 3), draw(gen, 2, 3.2));

    // empty subset in the middle of range
    EXPECT_EQ(pse_vector{}, draw(gen, 2, 2));

    // empty subset before first event
    EXPECT_EQ(pse_vector{}, draw(gen, 0, 0.05));

    // empty subset after last event
    EXPECT_EQ(pse_vector{}, draw(gen, 10, 11));

}

TEST(event_generators, poisson) {
    std::mt19937_64 G;

    time_type t0 = 0;
    time_type t1 = 10;
    time_type lambda = 10; // expect 10 events per ms
    cell_tag_type label = "label";
    cell_lid_type lid = 2;
    float weight = 42;

    event_generator gen = poisson_generator(label, weight, t0, lambda, G);
    gen.resolve_label([lid](const cell_local_label_type&) {return lid;});

    pse_vector int1 = as_vector(gen.events(0, t1));
    // Test that the output is sorted
    EXPECT_TRUE(std::is_sorted(int1.begin(), int1.end()));

    // Reset and generate the same sequence of events
    gen.reset();
    pse_vector int2 = as_vector(gen.events(0, t1));
    EXPECT_EQ(int1, int2);
}

