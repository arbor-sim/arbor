#pragma once

#include <array>
#include <atomic>

#include <util/debug.hpp>

namespace nest {
namespace mc {
namespace util {

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

    double_buffer(double_buffer&&) = delete;
    double_buffer(const double_buffer&) = delete;
    double_buffer& operator=(const double_buffer&) = delete;
    double_buffer& operator=(double_buffer&&) = delete;

    void exchange() {
        index_ ^= 1;
    }

    value_type& get() {
        return buffers_[index_];
    }

    const value_type& get() const {
        return buffers_[index_];
    }

    value_type& other() {
        return buffers_[other_index()];
    }

    const value_type& other() const {
        return buffers_[other_index()];
    }
};

} // namespace util
} // namespace mc
} // namespace nest
