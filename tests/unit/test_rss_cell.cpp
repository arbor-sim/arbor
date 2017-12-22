#include "../gtest.h"

#include <rss_cell.hpp>
#include <rss_cell_group.hpp>

#include "../simple_recipes.hpp"

using namespace arb;

using rss_recipe = homogeneous_recipe<cell_kind::regular_spike_source, rss_cell>;

TEST(rss_cell, basic_usage)
{
    constexpr time_type dt = 0.01; // dt is ignored by rss_cell_group::advance().

    // Use floating point times with an exact representation in order to avoid
    // rounding issues.
    rss_cell desc{0.125, 0.03125, 0.5};
    rss_cell_group sut({0}, rss_recipe(1u, desc));

    // No spikes in this time frame.
    epoch ep(0, 0.1);
    sut.advance(ep, dt, {});
    EXPECT_EQ(0u, sut.spikes().size());

    // Only on in this time frame
    sut.clear_spikes();
    ep.advance(0.127);
    sut.advance(ep, dt, {});
    EXPECT_EQ(1u, sut.spikes().size());

    // Reset cell group state.
    sut.reset();

    // Expect 12 spikes excluding the 0.5 end point.
    ep.advance(0.5);
    sut.advance(ep, dt, {});
    EXPECT_EQ(12u, sut.spikes().size());
}

TEST(rss_cell, poll_time_after_end_time)
{
    constexpr time_type dt = 0.01; // dt is ignored by rss_cell_group::advance().

    rss_cell desc{0.125, 0.03125, 0.5};
    rss_cell_group sut({0}, rss_recipe(1u, desc));

    // Expect 12 spikes in this time frame.
    sut.advance(epoch(0, 0.7), dt, {});
    EXPECT_EQ(12u, sut.spikes().size());

    // Now ask for spikes for a time slot already passed:
    // It should result in zero spikes because of the internal state!
    sut.clear_spikes();
    sut.advance(epoch(0, 0.2), dt, {});
    EXPECT_EQ(0u, sut.spikes().size());

    sut.reset();

    // Expect 12 excluding the 0.5
    sut.advance(epoch(0, 0.5), dt, {});
    EXPECT_EQ(12u, sut.spikes().size());
}

TEST(rss_cell, rate_bigger_then_epoch)
{
    constexpr time_type dt = 0.01; // dt is ignored by rss_cell_group::advance().

    rss_cell desc{ 0.0, 100.0, 1000.0 };
    rss_cell_group sut({ 0 }, rss_recipe(1u, desc));

    // take time steps of 10 ms
    for (time_type start = 0.0; start < 1000.0; start += 10) {
        sut.advance(epoch(start, start + 10.0), dt, {});
    }
    // We spike once every 100 ms so in 1000.0 ms we should have 10
    EXPECT_EQ(10u, sut.spikes().size());
}


TEST(rss_cell, cell_kind_correct)
{
    rss_cell desc{0.1, 0.01, 0.2};
    rss_cell_group sut({0}, rss_recipe(1u, desc));

    EXPECT_EQ(cell_kind::regular_spike_source, sut.get_cell_kind());
}
