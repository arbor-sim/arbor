#include "gtest.h"

#include <stimulus.hpp>

TEST(stimulus, i_clamp)
{
    using namespace nest::mc;

    // stimulus with delay 2, duration 0.5, amplitude 6.0
    i_clamp stim(2.0, 0.5, 6.0);

    EXPECT_EQ(stim.delay(), 2.0);
    EXPECT_EQ(stim.duration(), 0.5);
    EXPECT_EQ(stim.amplitude(), 6.0);

    // test that current only turned on in the half open interval
    // t \in [2, 2.5)
    EXPECT_EQ(stim.amplitude(0.0), 0.0);
    EXPECT_EQ(stim.amplitude(1.0), 0.0);
    EXPECT_EQ(stim.amplitude(2.0), 6.0);
    EXPECT_EQ(stim.amplitude(2.4999), 6.0);
    EXPECT_EQ(stim.amplitude(2.5), 0.0);

    // update: delay 1.0, duration 1.5, amplitude 3.0
    stim.set_delay(1.0);
    stim.set_duration(1.5);
    stim.set_amplitude(3.0);

    EXPECT_EQ(stim.delay(), 1.0);
    EXPECT_EQ(stim.duration(), 1.5);
    EXPECT_EQ(stim.amplitude(), 3.0);
}

