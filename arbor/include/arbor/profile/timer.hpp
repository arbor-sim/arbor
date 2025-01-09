#pragma once

#include <chrono>

// NOTE This is only in the public API as meter_manager is and has a private
//      member of tick_type.

namespace arb {
namespace profile {

using clock_type = std::chrono::steady_clock;
using tick_type = clock_type::time_point;

struct timer {
    constexpr static double scale = 1e-9; // ns -> s
    static inline tick_type tic() { return clock_type::now(); }
    static inline double toc(tick_type t) { return scale*std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - t).count(); }
};

} // namespace profile
} // namespace arb
