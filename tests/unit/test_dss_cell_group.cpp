#include "../gtest.h"

#include "dss_cell_description.hpp"
#include "dss_cell_group.hpp"
#include <util/unique_any.hpp>


using namespace  nest::mc;

TEST(dss_cell, constructor)
{
    std::vector<time_type> spikes;

    std::vector<util::unique_any> cell_descriptions(1);
    cell_descriptions[0] = util::unique_any(dss_cell_description(spikes));

    dss_cell_group sut({0}, cell_descriptions);
}

TEST(dss_cell, basic_usage)
{
    std::vector<time_type> spikes_to_emit;

    time_type spike_time = 0.1;
    spikes_to_emit.push_back(spike_time);

    std::vector<util::unique_any> cell_descriptions(1);
    cell_descriptions[0] = util::unique_any(dss_cell_description(spikes_to_emit));

    dss_cell_group sut({0}, cell_descriptions);

    // no spikes in this time frame
    sut.advance(0.09, 0.01);   // The dt (0,01) is not used

    auto spikes = sut.spikes();
    EXPECT_EQ(size_t(0), spikes.size());

    // only one in this time frame
    sut.advance(0.11, 0.01);
    spikes = sut.spikes();
    EXPECT_EQ(size_t(1), spikes.size());
    ASSERT_FLOAT_EQ(spike_time, spikes[0].time);

    // Clear the spikes after 'processing' them
    sut.clear_spikes();
    spikes = sut.spikes();
    EXPECT_EQ(size_t(0), spikes.size());

    // No spike to be emitted
    sut.advance(0.12, 0.01);
    spikes = sut.spikes();
    EXPECT_EQ(size_t(0), spikes.size());

    // Reset the internal state to null
    sut.reset();

    // Expect 10 excluding the 0.2
    sut.advance(0.2, 0.01);
    spikes = sut.spikes();
    EXPECT_EQ(size_t(1), spikes.size());
    ASSERT_FLOAT_EQ(spike_time, spikes[0].time);
}


TEST(dss_cell, cell_kind_correct)
{
    std::vector<time_type> spikes_to_emit;

    std::vector<util::unique_any> cell_descriptions(1);
    cell_descriptions[0] = util::unique_any(dss_cell_description(spikes_to_emit));

    dss_cell_group sut({0}, cell_descriptions);

    EXPECT_EQ(cell_kind::data_spike_source, sut.get_cell_kind());
}
