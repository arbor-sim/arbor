#include "../gtest.h"
#include "common.hpp"

#include <event_generator.hpp>

using namespace arb;
using pse = postsynaptic_spike_event;

namespace{
    auto compare=[](pse_vector expected, event_range r) {
        std::size_t i = 0;
        for (auto e: r) {
            if (i>=expected.size()) {
                FAIL() << "generated more events than expected";
            }
            EXPECT_EQ(expected[i], e);
            ++i;
        }
    };
}

TEST(event_generators, vector_backed) {
    std::vector<pse> in = {
        {{0, 0}, 0.0, 1.0},
        {{0, 0}, 1.0, 2.0},
        {{0, 0}, 1.0, 3.0},
        {{0, 0}, 1.5, 4.0},
        {{0, 0}, 2.3, 5.0},
        {{0, 0}, 3.0, 6.0},
        {{0, 0}, 3.5, 7.0},
    };

    vector_backed_generator gen(in);

    {   // all events in the range
        SCOPED_TRACE("all events");
        gen.reset();
        auto rng = gen.events(0, 4);
        compare(in, rng);
    }
    {   // a strict subset including the first event
        SCOPED_TRACE("subset with start");
        gen.reset();
        auto rng = gen.events(0, 3);
        pse_vector expected(in.begin(), in.begin()+6);
        compare(expected, rng);
    }
    {   // a strict subset including the last event
        SCOPED_TRACE("subset with last");
        gen.reset();
        auto rng = gen.events(1.5, 5);
        pse_vector expected(in.begin()+3, in.end());
        compare(expected, rng);
    }
}

