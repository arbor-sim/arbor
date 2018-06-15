#include <arbor/profile/clock.hpp>

namespace arb {
namespace profile {

template <typename Clock = default_clock>
struct timer<Clock> {
    static inline tick_type tic() {
        return Clock::now();
    }

    static inline tick_type toc(tick_type t) {
        return (Clock::now()-t)*Clock::seconds_per_tick();
    }
};

} // namespace profile
} // namespace arb
