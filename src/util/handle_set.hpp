#pragma once

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

namespace nest {
namespace mc {
namespace util {

template <typename Handle>
class handle_set {
public:
    using value_type = Handle;

    value_type acquire() {
        if (top==std::numeric_limits<Handle>::max()) {
            throw std::out_of_range("no more handles");
        }
        return top++;
    }

    // Pre-requisite: h is a handle returned by
    // `acquire`, which has not been subject
    // to a subsequent `release`.
    void release(value_type h) {
        if (h+1==top) {
            --top;
        }
    }

    // Release all handles.
    void clear() {
        top = 0;
    }

private:
    value_type top = 0;
};

} // namespace util
} // namespace mc
} // namespace nest

