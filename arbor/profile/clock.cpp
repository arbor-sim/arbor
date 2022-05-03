#define _POSIX_C_SOURCE 200809L
#include <time.h>

// Keep implementation out of header in order to avoid
// global namespace pollution from <time.h>.

#include <arbor/profile/clock.hpp>

namespace arb {
namespace profile {

inline tick_type posix_clock_gettime_ns(clockid_t clock) {
    timespec ts;
    if (clock_gettime(clock, &ts)) {
        return (unsigned long long)-1;
    }

    // According to SUS, we can assume tv_nsec is in [0, 1e9).

    tick_type seconds = ts.tv_sec;
    tick_type nanoseconds = 1000000000UL*seconds+(tick_type)ts.tv_nsec;

    return nanoseconds;
};

ARB_ARBOR_API tick_type posix_clock_gettime_monotonic_ns() {
    return posix_clock_gettime_ns(CLOCK_MONOTONIC);
}

} // namespace profile
} // namespace arb
