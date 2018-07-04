#include "../gtest.h"

#include <vector>

#include <arbor/time_sequence.hpp>

#include "util/rangeutil.hpp"

#include "common.hpp"

using namespace arb;

namespace{
    // Helper function that draws all samples in the half open interval
    // t ∈ [t0, t1) from a time_seq.
    std::vector<time_type> draw(time_seq& gen, time_type t0, time_type t1) {
        gen.reset();
        gen.advance(t0);
        std::vector<time_type> v;
        while (gen.front()<t1) {
            v.push_back(gen.front());
            gen.pop();
        }
        return v;
    }
}

TEST(time_seq, vector) {
    std::vector<time_type> times = {0.1, 1.0, 1.0, 1.5, 2.3, 3.0, 3.5, };

    vector_time_seq seq(times);

    // Test pop, next and reset.
    for (auto t: times) {
        EXPECT_EQ(t, seq.front());
        seq.pop();
    }
    seq.reset();
    for (auto t: times) {
        EXPECT_EQ(t, seq.front());
        seq.pop();
    }

    // The loop above should have drained all samples from seq, so we expect
    // that the front() time sample will be terminal_time.
    EXPECT_EQ(seq.front(), terminal_time);
}

TEST(time_seq, regular) {
    // make a regular generator that generates its first event at t=2ms and subsequent
    // events regularly spaced 0.5 ms apart.
    regular_time_seq seq(2, 0.5);

    // Test pop, next and reset.
    for (auto e:  {2.0, 2.5, 3.0, 3.5, 4.0, 4.5}) {
        EXPECT_EQ(e, seq.front());
        seq.pop();
    }
    seq.reset();
    for (auto e:  {2.0, 2.5, 3.0, 3.5, 4.0, 4.5}) {
        EXPECT_EQ(e, seq.front());
        seq.pop();
    }
    seq.reset();

    // Test advance()

    seq.advance(10.1);
    // next event greater ≥ 10.1 should be 10.5
    EXPECT_EQ(seq.front(), time_type(10.5));

    seq.advance(12);
    // next event greater ≥ 12   should be 12
    EXPECT_EQ(seq.front(), time_type(12));
}

// Test for rounding problems with large time values and the regular sequence
TEST(time_seq, regular_rounding) {
    // make a regular generator that generates its first time point at t=2ms
    // and subsequent times regularly spaced 0.5 ms apart.
    time_type t0 = 2.0;
    time_type dt = 0.5;

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
    time_seq seq = regular_time_seq(t0, dt);

    // Take the interval I_a: t ∈ [t0, t2)
    // And the two sub-interavls
    //      I_l: t ∈ [t0, t1)
    //      I_r: t ∈ [t1, t2)
    // Such that I_a = I_l ∪ I_r.
    // If we draw points from each interval then merge them, we expect same set
    // of points as when we draw from that large interval.
    std::vector<time_type> int_l = draw(seq, t0, t1);
    std::vector<time_type> int_r = draw(seq, t1, t2);
    std::vector<time_type> int_a = draw(seq, t0, t2);

    std::vector<time_type> int_merged = int_l;
    util::append(int_merged, int_r);

    EXPECT_TRUE(int_l.front() >= t0);
    EXPECT_TRUE(int_l.back()  <  t1);
    EXPECT_TRUE(int_r.front() >= t1);
    EXPECT_TRUE(int_r.back()  <  t2);
    EXPECT_EQ(int_a, int_merged);
    EXPECT_TRUE(std::is_sorted(int_a.begin(), int_a.end()));
}

TEST(time_seq, poisson) {
    std::mt19937_64 G;
    using pseq = poisson_time_seq<std::mt19937_64>;

    time_type t0 = 0;
    time_type t1 = 10;
    time_type lambda = 10; // expect 10 samples per ms

    pseq seq(G, t0, lambda);

    std::vector<time_type> int1;
    while (seq.front()<t1) {
        int1.push_back(seq.front());
        seq.pop();
    }
    // Test that the output is sorted
    EXPECT_TRUE(std::is_sorted(int1.begin(), int1.end()));

    // Reset and generate the same sequence of time points
    seq.reset();
    std::vector<time_type> int2;
    while (seq.front()<t1) {
        int2.push_back(seq.front());
        seq.pop();
    }

    // Assert that the same sequence was generated
    EXPECT_EQ(int1, int2);
}

// Test a poisson generator that has a tstop past which no samples
// should be generated.
TEST(time_seq, poisson_terminates) {
    std::mt19937_64 G;
    using pseq = poisson_time_seq<std::mt19937_64>;

    time_type t0 = 0;
    time_type t1 = 10;
    time_type t2 = 1e7; // pick a time far past the end of the interval [t0, t1)
    time_type lambda = 10; // expect 10 samples per ms

    // construct sequence with explicit end time t1
    pseq seq(G, t0, lambda, t1);

    std::vector<time_type> sequence;
    // pull samples off the sequence well past the end of the end time t1
    while (seq.front()<t2) {
        sequence.push_back(seq.front());
        seq.pop();
    }

    // the sequence should be exhausted
    EXPECT_EQ(seq.front(), terminal_time);

    // the last sample should be less than the end time
    EXPECT_TRUE(sequence.back()<t1);
}
