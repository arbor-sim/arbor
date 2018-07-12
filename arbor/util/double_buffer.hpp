#pragma once

#include <array>
#include <atomic>

#include <arbor/assert.hpp>
#include "arbor/execution_context.hpp"
namespace arb {
namespace util {

/// double buffer with thread safe exchange/flip operation.
template <typename T>
class double_buffer {
private:
    std::atomic<int> index_;
    std::array<T, 2> buffers_;

    int other_index() {
        return index_ ? 0 : 1;
    }

public:
    using value_type = T;

    double_buffer() :
        index_(0)
    {}

    /// remove the copy and move constructors which won't work with std::atomic
    double_buffer(double_buffer&&) = delete;
    double_buffer(const double_buffer&) = delete;
    double_buffer& operator=(const double_buffer&) = delete;
    double_buffer& operator=(double_buffer&&) = delete;

    /// flip the buffers in a thread safe manner
    /// n calls to exchange will always result in n flips
    void exchange() {
        // use operator^= which is overloaded by std::atomic<>
        index_ ^= 1;
    }

    /// get the current/front buffer
    value_type& get() {
        //return index_ ? buffer_1 : buffer_0;
        return buffers_[index_];
    }

    /// get the current/front buffer
    const value_type& get() const {
        //return index_ ? buffer_1 : buffer_0;
        return buffers_[index_];
    }

    /// get the back buffer
    value_type& other() {
        //return index_ ? buffer_0 : buffer_1;
        return buffers_[other_index()];
    }

    /// get the back buffer
    const value_type& other() const {
        //return index_ ? buffer_0 : buffer_1;
        return buffers_[other_index()];
    }

    void set_task_system(task_system_handle* ts) {
        buffers_[0].set_task_system(ts);
        buffers_[1].set_task_system(ts);
    }
};

} // namespace util
} // namespace arb
