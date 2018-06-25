#pragma once

typedef unsigned long long tick_type;

// Assuming POSIX monotonic clock is available; add
// feature test if we need to fall back to generic or
// other implementation.

namespace arb {
namespace profile {

tick_type posix_clock_gettime_monotonic_ns();

struct posix_clock_monotonic {
    static constexpr double seconds_per_tick() { return 1.e-9; }
    static unsigned long long now() {
        return posix_clock_gettime_monotonic_ns();
    }
};

using default_clock = posix_clock_monotonic;

} // namespace profile
} // namespace arb
