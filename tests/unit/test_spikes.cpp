#include "gtest.h"

#include <spike.hpp>
#include <spike_source.hpp>

struct cell_proxy {
    double voltage(nest::mc::segment_location loc) const {
        return v;
    }

    double v = -65.;
};

TEST(spikes, spike_detector)
{
    using namespace nest::mc;
    using detector_type = spike_detector<cell_proxy>;
    cell_proxy proxy;
    float threshold = 10.f;
    float t  = 0.f;
    float dt = 1.f;
    auto loc = segment_location(1, 0.1);

    auto detector = detector_type(proxy, loc, threshold, t);

    EXPECT_FALSE(detector.is_spiking());
    EXPECT_EQ(loc, detector.location());
    EXPECT_EQ(proxy.v, detector.v());
    EXPECT_EQ(t, detector.t());

    {
        t += dt;
        proxy.v = 0;
        auto spike = detector.test(proxy, t);
        EXPECT_FALSE(spike);

        EXPECT_FALSE(detector.is_spiking());
        EXPECT_EQ(loc, detector.location());
        EXPECT_EQ(proxy.v, detector.v());
        EXPECT_EQ(t, detector.t());
    }

    {
        t += dt;
        proxy.v = 20;
        auto spike = detector.test(proxy, t);

        EXPECT_TRUE(spike);
        EXPECT_EQ(spike.get(), 1.5);

        EXPECT_TRUE(detector.is_spiking());
        EXPECT_EQ(loc, detector.location());
        EXPECT_EQ(proxy.v, detector.v());
        EXPECT_EQ(t, detector.t());
    }

    {
        t += dt;
        proxy.v = 0;
        auto spike = detector.test(proxy, t);

        EXPECT_FALSE(spike);

        EXPECT_FALSE(detector.is_spiking());
        EXPECT_EQ(loc, detector.location());
        EXPECT_EQ(proxy.v, detector.v());
        EXPECT_EQ(t, detector.t());
    }
}

