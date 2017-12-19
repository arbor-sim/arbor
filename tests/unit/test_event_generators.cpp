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
        while (gen.next().time<t1) {
            v.push_back(gen.next());
            gen.pop();
        }
        return v;
    }
}

TEST(event_generators, vector_backed) {
    std::vector<pse> in = {
        {{0, 0}, 0.1, 1.0},
        {{0, 0}, 1.0, 2.0},
        {{0, 0}, 1.0, 3.0},
        {{0, 0}, 1.5, 4.0},
        {{0, 0}, 2.3, 5.0},
        {{0, 0}, 3.0, 6.0},
        {{0, 0}, 3.5, 7.0},
    };

    vector_backed_generator gen(in);

    // Test pop, next and reset.
    for (auto e: in) {
        EXPECT_EQ(e, gen.next());
        gen.pop();
    }
    gen.reset();
    for (auto e: in) {
        EXPECT_EQ(e, gen.next());
        gen.pop();
    }

    // The loop above should have drained all events from gen, so we expect
    // that the next() event will be the special terminal_pse event.
    EXPECT_EQ(gen.next(), terminal_pse());
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
        EXPECT_EQ(e, gen.next());
        gen.pop();
    }
    gen.reset();
    for (auto e:  expected({2.0, 2.5, 3.0, 3.5, 4.0, 4.5})) {
        EXPECT_EQ(e, gen.next());
        gen.pop();
    }
    gen.reset();

    // Test advance
    gen.advance(10.1);
    EXPECT_EQ(gen.next().time, time_type(10.5));
    gen.advance(12);
    EXPECT_EQ(gen.next().time, time_type(12));
}

// Test for rounding problems with large time values and the regular generator
TEST(event_generator, regular_rounding) {
    // make a regular generator that generates its first event at t=2ms and subsequent
    // events regularly spaced 0.5 ms apart.
    time_type t0 = 2.0;
    time_type dt = 0.5;
    cell_member_type target = {42, 3};
    float weight = 3.14;

    // Test for rounding problems with large time values.
    // To better understand why this is an issue, uncomment the following:
    //   float T = 1802667.0f, DT = 0.024999f;
    //   std::size_t N = std::floor(T/DT);
    //   std::cout << "T " << T << " DT " << DT << " N " << N
    //             << " T-N*DT " << T - (N*DT) << " P " << (T - (N*DT))/DT  << "\n";
    t0 = 1802667.0f;
    dt = 0.024999f;
    time_type int_len = 5*dt;
    time_type t1 = t0 + int_len;
    time_type t2 = t1 + int_len;
    auto gen = regular_generator(target, weight, t0, dt);

    // Take the interval I_a: t ∈ [t0, t2)
    // And the two sub-interavls
    //      I_l: t ∈ [t0, t1)
    //      I_r: t ∈ [t1, t2)
    // Such that I_a = I_l ∪ I_r.
    // If we draw events from each interval then merge them, we expect same set
    // of events as when we draw from that large interval.
    pse_vector int_l = draw(gen, t0, t1);
    pse_vector int_r = draw(gen, t1, t2);
    pse_vector int_a = draw(gen, t0, t2);

    pse_vector int_merged = int_l;
    util::append(int_merged, int_r);

    EXPECT_TRUE(int_l.front().time >= t0);
    EXPECT_TRUE(int_l.back().time  <  t1);
    EXPECT_TRUE(int_r.front().time >= t1);
    EXPECT_TRUE(int_r.back().time  <  t2);
    EXPECT_EQ(int_a, int_merged);
    EXPECT_TRUE(std::is_sorted(int_a.begin(), int_a.end()));
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

    seq_generator<pse_vector> gen(in);

    // Test pop, next and reset.
    for (auto e: in) {
        EXPECT_EQ(e, gen.next());
        gen.pop();
    }
    gen.reset();
    for (auto e: in) {
        EXPECT_EQ(e, gen.next());
        gen.pop();
    }
    // The loop above should have drained all events from gen, so we expect
    // that the next() event will be the special terminal_pse event.
    EXPECT_EQ(gen.next(), terminal_pse());

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
    while (gen.next().time<t1) {
        int1.push_back(gen.next());
        gen.pop();
    }
    // Test that the output is sorted
    EXPECT_TRUE(std::is_sorted(int1.begin(), int1.end()));

    // Reset and generate the same sequence of events
    gen.reset();
    pse_vector int2;
    while (gen.next().time<t1) {
        int2.push_back(gen.next());
        gen.pop();
    }

    // Assert that the same sequence was generated
    EXPECT_EQ(int1, int2);
}

// Test a poisson generator that has a tstop past which no events
// should be generated.
TEST(event_generators, poisson_terminates) {
    std::mt19937_64 G;
    using pgen = poisson_generator<std::mt19937_64>;

    time_type t0 = 0;
    time_type t1 = 10;
    time_type t2 = 1e7; // pick a time far past the end of the interval [t0, t1)
    time_type lambda = 10; // expect 10 events per ms
    cell_member_type target{4, 2};
    float weight = 42;
    // construct generator with explicit end time t1
    pgen gen(target, weight, G, t0, lambda, t1);

    pse_vector events;
    // pull events off the generator well past the end of the end time t1
    while (gen.next().time<t2) {
        events.push_back(gen.next());
        gen.pop();
    }

    // the generator should be exhausted
    EXPECT_EQ(gen.next(), terminal_pse());

    // the last event should be less than the end time
    EXPECT_TRUE(events.back().time<t1);
}
