#include "../gtest.h"
#include "common.hpp"

#include <event_generator.hpp>
#include <util/rangeutil.hpp>

using namespace arb;
using pse = postsynaptic_spike_event;

namespace{
    pse_vector draw(event_generator& gen, time_type t0, time_type t1) {
        gen.reset();
        gen.advance(t0);
        pse_vector v;
        while (gen.front().time<t1) {
            v.push_back(gen.front());
            gen.pop();
        }
        return v;
    }
}

TEST(event_generators, regular) {
    // make a regular generator that generates its first event at t=2ms and subsequent
    // events regularly spaced 0.5 ms apart.
    time_type t0 = 2.0;
    time_type dt = 0.5;
    cell_member_type target = {42, 3};
    float weight = 3.14;

    //regular_generator gen(t0, dt, target, weight);
    regular_generator gen(target, weight, t0, dt);

    // helper for building a set of 
    auto expected = [&] (std::vector<time_type> times) {
        pse_vector events;
        for (auto t: times) {
            events.push_back({target, t, weight});
        }
        return events;
    };

    // Test pop, next and reset.
    for (auto e:  expected({2.0, 2.5, 3.0, 3.5, 4.0, 4.5})) {
        EXPECT_EQ(e, gen.front());
        gen.pop();
    }
    gen.reset();
    for (auto e:  expected({2.0, 2.5, 3.0, 3.5, 4.0, 4.5})) {
        EXPECT_EQ(e, gen.front());
        gen.pop();
    }
    gen.reset();

    // Test advance
    gen.advance(10.1);
    EXPECT_EQ(gen.front().time, time_type(10.5));
    gen.advance(12);
    EXPECT_EQ(gen.front().time, time_type(12));
}

TEST(event_generators, seq) {
    std::vector<pse> in = {
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

    event_generator gen = seq_generator<pse_vector>(in);

    // Test pop, next and reset.
    for (auto e: in) {
        EXPECT_EQ(e, gen.front());
        gen.pop();
    }
    gen.reset();
    for (auto e: in) {
        EXPECT_EQ(e, gen.front());
        gen.pop();
    }
    // The loop above should have drained all events from gen, so we expect
    // that the front() event will be the special terminal_pse event.
    EXPECT_TRUE(is_terminal_pse(gen.front()));

    gen.reset();

    // Update of the input sequence, and run tests again to
    // verify that results reflect the new set of input events.
    in = {
        {{0, 0}, 1.5, 4.0},
        {{0, 0}, 2.3, 5.0},
        {{0, 0}, 3.0, 6.0},
        {{0, 0}, 3.5, 7.0},
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
    using pgen = poisson_generator<std::mt19937_64>;

    time_type t0 = 0;
    time_type t1 = 10;
    time_type lambda = 10; // expect 10 events per ms
    cell_member_type target{4, 2};
    float weight = 42;
    pgen gen(target, weight, G, t0, lambda);

    pse_vector int1;
    while (gen.front().time<t1) {
        int1.push_back(gen.front());
        gen.pop();
    }
    // Test that the output is sorted
    EXPECT_TRUE(std::is_sorted(int1.begin(), int1.end()));

    // Reset and generate the same sequence of events
    gen.reset();
    pse_vector int2;
    while (gen.front().time<t1) {
        int2.push_back(gen.front());
        gen.pop();
    }

    // Assert that the same sequence was generated
    EXPECT_EQ(int1, int2);
}

