#include "../gtest.h"

#include <sstream>
#include <string>
#include <vector>

#include <arbor/spike.hpp>
#include <aux/spike_emitter.hpp>

TEST(spike_emitter, formatting) {
    std::stringstream out;
    auto callback = aux::spike_emitter(out);

    std::vector<arb::spike> spikes = {
        { { 0, 0 }, 0.0 },
        { { 0, 0 }, 0.1 },
        { { 1, 0 }, 1.0 },
        { { 1, 0 }, 1.1 }
    };

    callback(spikes);

    std::string expected =
        "0 0.0000\n"
        "0 0.1000\n"
        "1 1.0000\n"
        "1 1.1000\n";

    EXPECT_EQ(expected, out.str());
}
