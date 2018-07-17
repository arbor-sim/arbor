#include "../gtest.h"

#include <arbor/schedule.hpp>
#include <arbor/spike.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/util/unique_any.hpp>

#include "spike_source_cell_group.hpp"

#include "../simple_recipes.hpp"

using namespace arb;
using ss_recipe = homogeneous_recipe<cell_kind::spike_source, spike_source_cell>;

// Test that a spike_source_cell_group identifies itself with the correct
// cell_kind enum value.
TEST(spike_source, cell_kind)
{
    ss_recipe rec(1u, spike_source_cell{explicit_schedule({})});
    spike_source_cell_group group({0}, rec);

    EXPECT_EQ(cell_kind::spike_source, group.get_cell_kind());
}

static std::vector<time_type> as_vector(std::pair<const time_type*, const time_type*> ts) {
    return std::vector<time_type>(ts.first, ts.second);
}

static std::vector<time_type> spike_times(const std::vector<spike>& evs) {
    std::vector<time_type> ts;
    for (auto& s: evs) {
        ts.push_back(s.time);
    }
    return ts;
}

// Test that a spike_source_cell_group produces a sequence of spikes with spike
// times corresponding to the underlying time_seq.
TEST(spike_source, matches_time_seq)
{
    auto test_seq = [](schedule seq) {
        ss_recipe rec(1u, spike_source_cell{seq});
        spike_source_cell_group group({0}, rec);

        // epoch ending at 10ms
        epoch ep(0, 10);
        group.advance(ep, 1, {});
        EXPECT_EQ(spike_times(group.spikes()), as_vector(seq.events(0, 10)));

        group.clear_spikes();

        // advance to 20 ms and repeat
        ep.advance(20);
        group.advance(ep, 1, {});
        EXPECT_EQ(spike_times(group.spikes()), as_vector(seq.events(10, 20)));
    };

    std::mt19937_64 G;
    test_seq(regular_schedule(0, 1));
    test_seq(poisson_schedule(10, G));   // produce many spikes in each interval
    test_seq(poisson_schedule(1e-6, G)); // very unlikely to produce any spikes in either interval
}

// Test that a spike_source_cell_group will produce the same sequence of spikes
// after being reset.
TEST(spike_source, reset)
{
    auto test_seq = [](schedule seq) {
        ss_recipe rec(1u, spike_source_cell{seq});
        spike_source_cell_group group({0}, rec);

        // Advance for 10 ms and store generated spikes in spikes1.
        epoch ep(0, 10);
        group.advance(ep, 1, {});
        auto spikes1 = group.spikes();

        // Reset the model, then advance again to 10 ms, and store the
        // generated spikes in spikes2.
        group.reset();
        group.advance(ep, 1, {});
        auto spikes2 = group.spikes();

        // Check that the same spikes were generated in each case.
        EXPECT_EQ(spikes1, spikes2);
    };

    std::mt19937_64 G;
    test_seq(regular_schedule(0, 1));
    test_seq(poisson_schedule(10, G));   // produce many spikes in each interval
    test_seq(poisson_schedule(1e-6, G)); // very unlikely to produce any spikes in either interval
}

// Test that a spike_source_cell_group will produce the expected
// output when the underlying time_seq is finite.
TEST(spike_source, exhaust)
{
    // This test assumes that seq will exhaust itself before t=10 ms.
    auto test_seq = [](schedule seq) {
        ss_recipe rec(1u, spike_source_cell{seq});
        spike_source_cell_group group({0}, rec);

        // epoch ending at 10ms
        epoch ep(0, 10);
        group.advance(ep, 1, {});
        EXPECT_EQ(spike_times(group.spikes()), as_vector(seq.events(0, 10)));

        // Check that the last spike was before the end of the epoch.
        EXPECT_LT(group.spikes().back().time, time_type(10));
    };

    test_seq(regular_schedule(0, 1, 5));
    test_seq(explicit_schedule({0.3, 2.3, 4.7}));
}
