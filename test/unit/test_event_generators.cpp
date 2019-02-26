#include "../gtest.h"

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

TEST(event_generators, regular) {
    // make a regular generator that generates its first event at t=2ms and subsequent
    // events regularly spaced 0.5 ms apart.
    time_type t0 = 2.0;
    time_type dt = 0.5;
    cell_member_type target = {42, 3};
    float weight = 3.14;

    event_generator gen = regular_generator(target, weight, t0, dt);

    // helper for building a set of 
    auto expected = [&] (std::vector<time_type> times) {
        pse_vector events;
        for (auto t: times) {
            events.push_back({target, t, weight});
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

TEST(event_generators, seq) {
    pse_vector in = {
        {{0, 0}, 0.1, 1.0},
        {{0, 0}, 1.0, 2.0},
        {{0, 0}, 1.0, 3.0},
        {{0, 0}, 1.5, 4.0},
        {{0, 0}, 2.3, 5.0},
        {{0, 0}, 3.0, 6.0},
        {{0, 0}, 3.5, 7.0},
    };

    auto events = [&in] (int b, int e) {
        return pse_vector(in.begin()+b, in.begin()+e);
    };

    event_generator gen = explicit_generator(in);
    EXPECT_EQ(in, as_vector(gen.events(0, 100.)));
    gen.reset();
    EXPECT_EQ(in, as_vector(gen.events(0, 100.)));
    gen.reset();

    // Check reported sub-intervals against a smaller set of events.
    in = {
        {{0, 0}, 1.5, 4.0},
        {{0, 0}, 2.3, 5.0},
        {{0, 0}, 3.0, 6.0},
        {{0, 0}, 3.5, 7.0},
    };
    gen = explicit_generator(in);

    auto draw = [](event_generator& gen, time_type t0, time_type t1) {
        gen.reset();
        return as_vector(gen.events(t0, t1));
    };

    // a range that includes all the events
    EXPECT_EQ(in, draw(gen, 0, 4));

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
    cell_member_type target{4, 2};
    float weight = 42;

    event_generator gen = poisson_generator(target, weight, t0, lambda, G);

    pse_vector int1 = as_vector(gen.events(0, t1));
    // Test that the output is sorted
    EXPECT_TRUE(std::is_sorted(int1.begin(), int1.end()));

    // Reset and generate the same sequence of events
    gen.reset();
    pse_vector int2 = as_vector(gen.events(0, t1));
    EXPECT_EQ(int1, int2);
}

