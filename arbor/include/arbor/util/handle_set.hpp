#pragma once

#include <atomic>
#include <stdexcept>

/*
 * Manage a set of integer-valued handles.
 *
 * An error is thrown if no unused handle value is available on acquire.
 *
 * Simple implementation below does not try hard to reuse released
 * handles; smarter versions to be implemented as required.
 */

namespace arb {
namespace util {

class handle_set {
public:
    using value_type = std::size_t;

    value_type acquire() {
        auto nxt = top_.fetch_add(1);
        // We would run into UB _next_ time, so die now.
        if (top_ == std::numeric_limits<value_type>::max()) {
            throw std::out_of_range("no more handles");
        }
        return nxt;
    }

    // Pre-requisite: h is a handle returned by `acquire`, which has not been
    // subject to a subsequent `release`.
    void release(value_type h) {
        // _if_ this was the last handle to be acquire, release it.
        value_type ex = h + 1;
        top_.compare_exchange_strong(ex, ex - 1);
        // if not, continue as if nothing happened.
    }

    // Release all handles.
    void clear() { top_.store(0); }

private:
    std::atomic<value_type> top_ = 0;
};

} // namespace util
} // namespace arb

