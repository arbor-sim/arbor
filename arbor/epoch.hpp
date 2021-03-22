#pragma once

#include <cstdint>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>

namespace arb {

// An epoch describes an integration interval within an ongoing simulation.
// Epochs within a simulation are sequentially numbered by id, and each
// describe a half open time interval [t0, t1), where t1 for one epoch
// is t0 for the next.
//
// At the end of an epoch the simulation state corresponds to the time t1,
// save for any pending events, which will be delivered at the beginning of
// the next epoch.

struct epoch {
    std::ptrdiff_t id = -1;
    time_type t0 = 0, t1 = 0;

    epoch() = default;
    epoch(std::ptrdiff_t id, time_type t0, time_type t1):
        id(id), t0(t0), t1(t1) {}

    operator bool() const {
        arb_assert(id>=-1);
        return id>=0;
    }

    bool empty() const {
        return t1<=t0;
    }

    time_type duration() const {
        return t1-t0;
    }

    void advance_to(time_type next_t1) {
        t0 = t1;
        t1 = next_t1;
        ++id;
    }

    void reset() {
        *this = epoch();
    }

};

} // namespace arb
