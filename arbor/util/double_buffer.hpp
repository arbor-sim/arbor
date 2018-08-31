#pragma once

#include <atomic>
#include <vector>

#include <arbor/assert.hpp>

namespace arb {
namespace util {

/// double buffer with thread safe exchange/flip operation.
template <typename T>
class double_buffer {
private:
    std::atomic<int> index_;
    std::vector<T> buffers_;

    int other_index() {
        return index_ ? 0 : 1;
    }

public:
    using value_type = T;

    double_buffer() :
        index_(0), buffers_(2)
    {}

    double_buffer(T l, T r): index_(0) {
        buffers_.reserve(2);
        buffers_.push_back(std::move(l));
        buffers_.push_back(std::move(r));
    }

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
        return buffers_[index_];
    }

    /// get the current/front buffer
    const value_type& get() const {
        return buffers_[index_];
    }

    /// get the back buffer
    value_type& other() {
        return buffers_[other_index()];
    }

    /// get the back buffer
    const value_type& other() const {
        return buffers_[other_index()];
    }
};

} // namespace util
} // namespace arb
