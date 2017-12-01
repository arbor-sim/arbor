#include "../gtest.h"
#include "common.hpp"

#include <event_generator.hpp>
#include <util/rangeutil.hpp>

using namespace arb;
using pse = postsynaptic_spike_event;

namespace{
    auto compare=[](pse_vector expected, event_range r) {
        //std::cout << "EXPECTED: "; for (auto e: expected) std::cout << e << " "; std::cout << "\n";
        //std::cout << "RESULT  : "; for (auto e: r) std::cout << e << " "; std::cout << "\n";
        std::size_t i = 0;
        for (auto e: r) {
            if (i>=expected.size()) {
                FAIL() << "generated more events than expected";
            }
            EXPECT_EQ(expected[i], e);
            ++i;
        }
        EXPECT_EQ(expected.size(), i) << "generated less events than expected";
    };
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

    auto events = [&in] (int b, int e) {
        return pse_vector(in.begin()+b, in.begin()+e);
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
    gen.reset();

    // Test the selection of events over intervals.
    // Note: the vector_back storage does not require a reset because
    // calls to events() always search the vector storage for the first
    // event in the interval.

    // a range that includes all the events
    {
        SCOPED_TRACE("all events");
        compare(in, gen.events(0, 4));
    }

    // a strict subset including the first event
    {
        SCOPED_TRACE("subset with start");
        compare(events(0, 5), gen.events(0, 3));
    }

    // a strict subset including the last event
    {
        SCOPED_TRACE("subset with last");
        compare(events(3, 7), gen.events(1.5, 5));
    }

    // subset that excludes first and last entries
    {
        SCOPED_TRACE("subset");
        compare(events(3, 5), gen.events(1.5, 3));
    }

    // empty subset in the middle of range
    {
        SCOPED_TRACE("empty subset");
        compare({}, gen.events(2, 2));
    }

    // empty subset before first event
    {
        SCOPED_TRACE("empty early");
        compare({}, gen.events(0, 0.05));
    }

    // empty subset after last event
    {
        SCOPED_TRACE("empty late");
        compare({}, gen.events(10, 11));
    }
}

TEST(event_generators, regular) {
    // make a regular generator that generates its first event at t=2ms and subsequent
    // events regularly spaced 0.5 ms apart.
    time_type t0 = 2.0;
    time_type dt = 0.5;
    cell_member_type target = {42, 3};
    float weight = 3.14;

    regular_generator gen(t0, dt, target, weight);

    // helper for building a set of 
    auto expected = [&] (std::vector<time_type> times) {
        pse_vector events;
        for (auto t: times) events.push_back({target, t, weight});
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

    {
        SCOPED_TRACE("before t0");
        compare(expected({2.0, 2.5, 3.0, 3.5, 4.0, 4.5}), gen.events(2, 5));
    }
    {
        SCOPED_TRACE("starts after t0");
        compare(expected({3.0, 3.5, 4.0, 4.5, 5.0}), gen.events(2.9, 5.1));
    }
    {
        SCOPED_TRACE("empty before t0");
        compare(expected({}), gen.events(0, 2));
    }
    {
        SCOPED_TRACE("empty after t0");
        compare(expected({}), gen.events(2.2, 2.5));
    }

    // Test for rounding problems with large time values.
    // To better understand why this is an issue, uncomment the following:
    //   float T = 1802667.0f, DT = 0.024999f;
    //   std::size_t N = std::floor(T/DT);
    //   std::cout << "T " << T << " DT " << DT << " N " << N
    //              << " T-N*DT " << T - (N*DT) << " P " << (T - (N*DT))/DT  << "\n";
    t0 = 1802667.0f;
    dt = 0.024999f;
    time_type int_len = 5*dt;
    time_type t1 = t0 + int_len;
    time_type t2 = t1 + int_len;
    gen = regular_generator(t0, dt, target, weight);

    // Take the interval I_a: t ∈ [t0, t2)
    // And the two sub-interavls
    //      I_l: t ∈ [t0, t1)
    //      I_r: t ∈ [t1, t2)
    // Such that I_a = I_l ∪ I_r.
    // If we draw events from each interval then merge them, we expect same set
    // of events as when we draw from that large interval.
    pse_vector int_l = util::assign_from(gen.events(t0, t1));
    pse_vector int_r = util::assign_from(gen.events(t1, t2));
    pse_vector int_a = util::assign_from(gen.events(t0, t2));

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
    gen.reset();

    // Test the selection of events over intervals.
    // Note: the vector_back storage does not require a reset because
    // calls to events() always search the vector storage for the first
    // event in the interval.

    {   // a range that includes all the events
        SCOPED_TRACE("all events");
        compare(in, gen.events(0, 4));
    }

    {   // a strict subset including the first event
        SCOPED_TRACE("subset with start");
        compare(events(0, 5), gen.events(0, 3));
    }

    {   // a strict subset including the last event
        SCOPED_TRACE("subset with last");
        compare(events(3, 7), gen.events(1.5, 5));
    }

    {   // subset that excludes first and last entries
        SCOPED_TRACE("subset");
        compare(events(3, 5), gen.events(1.5, 3));
    }

    {   // empty subset in the middle of range
        SCOPED_TRACE("empty subset");
        compare({}, gen.events(2, 2));
    }

    {   // empty subset before first event
        SCOPED_TRACE("empty early");
        compare({}, gen.events(0, 0.05));
    }

    {   // empty subset after last event
        SCOPED_TRACE("empty late");
        compare({}, gen.events(10, 11));
    }

    // Update of the input sequence, and run tests again to
    // verify that results reflect the new set of input events.
    in = {
        {{0, 0}, 1.5, 4.0},
        {{0, 0}, 2.3, 5.0},
        {{0, 0}, 3.0, 6.0},
        {{0, 0}, 3.5, 7.0},
    };

    {   // a range that includes all the events
        SCOPED_TRACE("all events");
        compare(in, gen.events(0, 4));
    }

    {   // a strict subset including the first event
        SCOPED_TRACE("subset with start");
        compare(events(0, 2), gen.events(0, 3));
    }

    {   // a strict subset including the last event
        SCOPED_TRACE("subset with last");
        compare(events(2, 4), gen.events(3.0, 5));
    }

    {   // subset that excludes first and last entries
        SCOPED_TRACE("subset");
        compare(events(1, 3), gen.events(2, 3.2));
    }

    {   // empty subset in the middle of range
        SCOPED_TRACE("empty subset");
        compare({}, gen.events(2, 2));
    }

    {   // empty subset before first event
        SCOPED_TRACE("empty early");
        compare({}, gen.events(0, 0.05));
    }

    {   // empty subset after last event
        SCOPED_TRACE("empty late");
        compare({}, gen.events(10, 11));
    }
}

TEST(event_generators, poisson) {
    std::mt19937_64 G;
    using pgen = poisson_generator<std::mt19937_64>;

    time_type t0 = 0;
    time_type dt = 0.1;
    cell_member_type target{4, 2};
    float weight = 42;
    pgen gen(t0, dt, target, weight, G);

    pse_vector int_all;
    util::append(int_all, gen.events(0, 10));
    gen.reset();
    pse_vector int_mrg;
    util::append(int_mrg, gen.events(0, 1));
    util::append(int_mrg, gen.events(1, 2));
    util::append(int_mrg, gen.events(2, 3));
    util::append(int_mrg, gen.events(3, 4));
    util::append(int_mrg, gen.events(4, 5));
    util::append(int_mrg, gen.events(5, 6));
    util::append(int_mrg, gen.events(6, 7));
    util::append(int_mrg, gen.events(7, 8));
    util::append(int_mrg, gen.events(8, 9));
    util::append(int_mrg, gen.events(9, 10));

    EXPECT_EQ(int_mrg, int_all);
}
