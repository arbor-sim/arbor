#include "../gtest.h"

#include "fs_cell.hpp"

using namespace  nest::mc;

TEST(fs_cell, constructor)
{
    fs_cell test(0.0, 0.01, 1.0);
}


TEST(fs_cell, correct_usage)
{
    fs_cell sut(0.1, 0.01, 0.2);


    // no spikes in this time frame
    auto spikes = sut.spikes_until(0.09);
    EXPECT_EQ(size_t(0), spikes.size());

    //only on in this time frame
    spikes = sut.spikes_until(0.11);
    EXPECT_EQ(size_t(1), spikes.size());

    sut.reset();

    // Expect 10 excluding the 0.2
    spikes = sut.spikes_until(0.2);
    EXPECT_EQ(size_t(10), spikes.size());
}



    // test construction
