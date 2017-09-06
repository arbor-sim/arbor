#pragma once

#include <algorithm>

#include <memory/allocator.hpp>
#include "stack_common.hpp"

namespace nest {
namespace mc {
namespace gpu {

// A simple stack data structure for the GPU.
//
// Provides a host side interface for
//  * construction and destrutcion
//  * reading the values stored on the stack
//  * resetting the stack to an empty state
//  * querying the size and capacity of the stack
// Provides a device side interface for
//  * cooperative grid level push_back
//  * querying the size and capacity of the stack
//
// It is designed to be initialized empty with a given capacity on the host,
// updated by device kernels, and periodically read and reset from the host side.
template <typename T>
class stack {
    using value_type = T;
    template <typename U>
    using allocator = memory::managed_allocator<U>;

    using base_type = stack_base<value_type>;
    base_type* base_;

public:
    stack& operator==(const stack& other) = delete;
    stack(const stack& other) = delete;

    stack(stack&& other) {
        std::swap(base_, other.base_);
    }

    stack& operator=(stack&& other) {
        std::swap(base_, other.base_);
        return *this;
    }

    stack(unsigned capacity) {
        base_ = allocator<base_type>().allocate(1);
        base_->capacity = capacity;
        base_->size = 0;
        base_->data = allocator<value_type>().allocate(capacity);
    }

    stack(): stack(0) {}

    ~stack() {
        if (base_) {
            allocator<value_type>().deallocate(base_->data, base_->capacity);
            allocator<base_type>().deallocate(base_, 1);
        }
    }

    void clear() {
        base_->size = 0;
    }

    // The number of items that have been pushed back on the stack.
    // size may exceed capacity, which indicates that the caller attempted
    // to push back more values than there was space to store.
    unsigned size() const {
        return base_->size;
    }

    // The maximum number of items that can be stored in the stack.
    unsigned capacity() const {
        return base_->capacity;
    }

    base_type& base() {
        return *base_;
    }

    value_type& operator[](unsigned i) {
        EXPECTS(i<base_->size && i<base_->capacity);
        return base_->data[i];
    }

    value_type& operator[](unsigned i) const {
        EXPECTS(i<base_->size && i<base_->capacity);
        return base_->data[i];
    }

    value_type* begin() {
        return base_->data;
    }
    const value_type* begin() const {
        return base_->data;
    }

    value_type* end() {
        // Take care of the case where size>capacity.
        return base_->data + std::min(base_->size, base_->capacity);
    }
    const value_type* end() const {
        // Take care of the case where size>capacity.
        return base_->data + std::min(base_->size, base_->capacity);
    }
};

} // namespace gpu
} // namespace mc
} // namespace nest
