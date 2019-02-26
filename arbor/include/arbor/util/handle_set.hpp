#pragma once

#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

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

template <typename Handle>
class handle_set {
public:
    using value_type = Handle;

    value_type acquire() {
        lock_guard lock(mex_);

        if (top_==std::numeric_limits<Handle>::max()) {
            throw std::out_of_range("no more handles");
        }
        return top_++;
    }

    // Pre-requisite: h is a handle returned by
    // `acquire`, which has not been subject
    // to a subsequent `release`.
    void release(value_type h) {
        lock_guard lock(mex_);

        if (h+1==top_) {
            --top_;
        }
    }

    // Release all handles.
    void clear() {
        lock_guard lock(mex_);

        top_ = 0;
    }

private:
    value_type top_ = 0;

    using lock_guard = std::lock_guard<std::mutex>;
    std::mutex mex_;
};

} // namespace util
} // namespace arb

