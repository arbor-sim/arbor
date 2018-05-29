#include "../gtest.h"

#include <spike_source_cell_group.hpp>
#include <time_sequence.hpp>
#include <util/unique_any.hpp>

#include "../simple_recipes.hpp"

using namespace arb;
using ss_recipe = homogeneous_recipe<cell_kind::spike_source, time_seq>;

// Test that a spike_source_cell_group identifies itself with the correct
// cell_kind enum value.
TEST(spike_source, cell_kind)
{
    ss_recipe rec(1u, vector_time_seq({}));
    spike_source_cell_group group({0}, rec);

    EXPECT_EQ(cell_kind::spike_source, group.get_cell_kind());
}

// Test that a spike_source_cell_group will produce the same sequence of spikes
// with times corresponding to the underlying time_seq.
TEST(spike_source, matches_time_seq)
{
    auto test_seq = [](arb::time_seq seq) {
        ss_recipe rec(1u, seq);
        spike_source_cell_group group({0}, rec);

        // epoch ending at 10ms
        epoch ep(0, 10);
        group.advance(ep, 1, {});
        for (auto s: group.spikes()) {
            EXPECT_EQ(s.time, seq.next());
            seq.pop();
        }
        EXPECT_TRUE(seq.next()>=ep.tfinal);
        group.clear_spikes();

        // advance to 20 ms and repeat
        ep.advance(20);
        group.advance(ep, 1, {});
        for (auto s: group.spikes()) {
            EXPECT_EQ(s.time, seq.next());
            seq.pop();
        }
        EXPECT_TRUE(seq.next()>=ep.tfinal);
    };

    test_seq(arb::regular_time_seq(0,1));
    std::mt19937_64 G;
    using pseq = arb::poisson_time_seq<std::mt19937_64>;
    test_seq(pseq(G, 0., 10));    // produce many spikes in each interval
    test_seq(pseq(G, 0., 1e-6)); // very unlikely to produce any spikes in either interval
}

// Test that a spike_source_cell_group will produce the same sequence of spikes
// after buing reset.
TEST(spike_source, reset)
{
    auto test_seq = [](arb::time_seq seq) {
        ss_recipe rec(1u, seq);
        spike_source_cell_group group({0}, rec);

        // epoch ending at 10ms
        epoch ep(0, 10);
        group.advance(ep, 1, {});
        auto spikes = group.spikes();
        group.reset();
        EXPECT_TRUE(group.spikes().empty());

        // advance to 20 ms and repeat
        group.advance(ep, 1, {});
        EXPECT_EQ(spikes.size(), group.spikes().size());
        EXPECT_EQ(spikes, group.spikes());
    };

    test_seq(arb::regular_time_seq(0,1));
    std::mt19937_64 G;
    using pseq = arb::poisson_time_seq<std::mt19937_64>;
    test_seq(pseq(G, 0., 10));    // produce many spikes in each interval
    test_seq(pseq(G, 0., 1e-6)); // very unlikely to produce any spikes in either interval
}

// Test that a spike_source_cell_group will produce the expected
// output when the underlying time_seq is finite.
TEST(spike_source, exhaust)
{
    // This test assumes that seq will exhaust itself before t=10 ms.
    auto test_seq = [](arb::time_seq seq) {
        ss_recipe rec(1u, seq);
        spike_source_cell_group group({0}, rec);

        // epoch ending at 10ms
        epoch ep(0, 10);
        group.advance(ep, 1, {});
        auto spikes = group.spikes();
        for (auto s: group.spikes()) {
            EXPECT_EQ(s.time, seq.next());
            seq.pop();
        }
        // assert that the next value in the sequence is marked as time_max
        EXPECT_EQ(seq.next(), arb::max_time);
        // assert that the last spike was before the end of the epoch
        EXPECT_LT(spikes.back().time, 10);

    };

    test_seq(arb::regular_time_seq(0,1,5));
    std::mt19937_64 G;
    using pseq = arb::poisson_time_seq<std::mt19937_64>;
    test_seq(pseq(G, 0., 10, 5));    // produce many spikes in each interval
}
