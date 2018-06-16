#pragma once

#include <cstdint>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>

namespace arb {

// Information about a current time integration epoch.
// Each epoch has an integral id, that is incremented for successive epochs.
// Time is divided as follows, where tfinal_i is tfinal for epoch i:
//
// epoch_0 :             t < tfinal_0
// epoch_1 : tfinal_0 <= t < tfinal_1
// epoch_2 : tfinal_1 <= t < tfinal_2
// epoch_3 : tfinal_2 <= t < tfinal_3
//
// At the end of epoch_i the solution is at tfinal_i, however events that are
// due for delivery at tfinal_i are not delivered until epoch_i+1.

struct epoch {
    std::size_t id=0;
    time_type tfinal=0;

    epoch() = default;

    epoch(std::size_t id, time_type tfinal): id(id), tfinal(tfinal) {}

    void advance(time_type t) {
        arb_assert(t>=tfinal);
        tfinal = t;
        ++id;
    }
};

} // namespace arb
