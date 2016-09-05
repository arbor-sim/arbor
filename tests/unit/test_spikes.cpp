#include "gtest.h"

#include <spike.hpp>
#include <spike_source.hpp>

struct cell_proxy {
    using detector_handle = int;
    double detector_voltage(detector_handle) const {
        return v;
    }

    double v = -65.;
};

TEST(spikes, spike_detector)
{
    using namespace nest::mc;
    using detector_type = spike_detector<cell_proxy>;
    using detector_handle = cell_proxy::detector_handle;

    cell_proxy proxy;
    float threshold = 10.f;
    float t  = 0.f;
    float dt = 1.f;
    detector_handle handle{};

    auto detector = detector_type(proxy, handle, threshold, t);

    EXPECT_FALSE(detector.is_spiking());
    EXPECT_EQ(proxy.v, detector.v());
    EXPECT_EQ(t, detector.t());

    {
        t += dt;
        proxy.v = 0;
        auto spike = detector.test(proxy, t);
        EXPECT_FALSE(spike);

        EXPECT_FALSE(detector.is_spiking());
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
        EXPECT_EQ(proxy.v, detector.v());
        EXPECT_EQ(t, detector.t());
    }

    {
        t += dt;
        proxy.v = 0;
        auto spike = detector.test(proxy, t);

        EXPECT_FALSE(spike);

        EXPECT_FALSE(detector.is_spiking());
        EXPECT_EQ(proxy.v, detector.v());
        EXPECT_EQ(t, detector.t());
    }
}

