#include "../gtest.h"

#include <dss_cell_description.hpp>
#include <dss_cell_group.hpp>
#include <util/unique_any.hpp>

#include "../simple_recipes.hpp"

using namespace arb;

using dss_recipe = homogeneous_recipe<cell_kind::data_spike_source, dss_cell_description>;

TEST(dss_cell, basic_usage)
{
    const time_type spike_time = 0.1;
    dss_recipe rec(1u, dss_cell_description({spike_time}));
    dss_cell_group sut({0}, rec);

    // No spikes in this time frame.
    time_type dt = 0.01; // (note that dt is ignored in dss_cell_group).
    sut.advance(0.09, dt);

    auto spikes = sut.spikes();
    EXPECT_EQ(0u, spikes.size());

    // Only one in this time frame.
    sut.advance(0.11, 0.01);
    spikes = sut.spikes();
    EXPECT_EQ(1u, spikes.size());
    ASSERT_FLOAT_EQ(spike_time, spikes[0].time);

    // Clear the spikes after 'processing' them.
    sut.clear_spikes();
    spikes = sut.spikes();
    EXPECT_EQ(0u, spikes.size());

    // No spike to be emitted.
    sut.advance(0.12, dt);
    spikes = sut.spikes();
    EXPECT_EQ(0u, spikes.size());

    // Reset the internal state.
    sut.reset();

    // Expect to have the one spike again after reset.
    sut.advance(0.2, dt);
    spikes = sut.spikes();
    EXPECT_EQ(1u, spikes.size());
    ASSERT_FLOAT_EQ(spike_time, spikes[0].time);
}


TEST(dss_cell, cell_kind_correct)
{
    const time_type spike_time = 0.1;
    dss_recipe rec(1u, dss_cell_description({spike_time}));
    dss_cell_group sut({0}, rec);

    EXPECT_EQ(cell_kind::data_spike_source, sut.get_cell_kind());
}
