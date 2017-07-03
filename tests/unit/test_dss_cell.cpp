#include "../gtest.h"

#include "dss_cell.hpp"

using namespace  nest::mc;

TEST(dss_cell, constructor)
{
    std::vector<time_type> spikes;

    auto descr = dss_cell::dss_cell_description(spikes);

    dss_cell sut(descr);
}

TEST(dss_cell, basic_usage)
{
    std::vector<time_type> spikes_to_emit;

    time_type spike_time = 0.1;
    spikes_to_emit.push_back(spike_time);
    ASSERT_DOUBLE_EQ(spike_time, spikes_to_emit[0]);


    auto descr = dss_cell::dss_cell_description(spikes_to_emit);
    dss_cell sut(descr);

    // no spikes in this time frame
    std::vector<time_type> spikes = sut.spikes_until(0.09);
    EXPECT_EQ(size_t(0), spikes.size());

    // only on in this time frame
    spikes = sut.spikes_until(0.11);
    EXPECT_EQ(size_t(1), spikes.size());
    ASSERT_FLOAT_EQ(spike_time, spikes[0]);

    // only one spike to be emitted
    spikes = sut.spikes_until(0.12);
    EXPECT_EQ(size_t(0), spikes.size());

    // Reset the internal state to null
    sut.reset();

    // Expect 10 excluding the 0.2
    spikes = sut.spikes_until(0.2);
    EXPECT_EQ(size_t(1), spikes.size());
    ASSERT_FLOAT_EQ(spike_time, spikes[0]);
}


TEST(dss_cell, cell_kind_correct)
{
    std::vector<time_type> spikes_to_emit;

    auto descr = dss_cell::dss_cell_description(spikes_to_emit);
    dss_cell sut(descr);

    EXPECT_EQ(cell_kind::data_spike_source, sut.get_cell_kind());
}
