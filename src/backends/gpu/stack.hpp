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

    using storage_type = stack_storage<value_type>;
    storage_type* storage_;

    storage_type* create_storage(unsigned n) {
        auto p = allocator<storage_type>().allocate(1);
        p->capacity = n;
        p->size = 0;
        p->data = allocator<value_type>().allocate(n);
        return p;
    }

public:
    stack& operator=(const stack& other) = delete;
    stack(const stack& other) = delete;

    stack(): storage_(create_storage(0)) {}

    stack(stack&& other): storage_(create_storage(0)) {
        std::swap(storage_, other.storage_);
    }

    stack& operator=(stack&& other) {
        std::swap(storage_, other.storage_);
        return *this;
    }

    explicit stack(unsigned capacity): storage_(create_storage(capacity)) {}

    ~stack() {
        allocator<value_type>().deallocate(storage_->data, storage_->capacity);
        allocator<storage_type>().deallocate(storage_, 1);
    }

    void clear() {
        storage_->size = 0u;
    }

    // The number of items that have been pushed back on the stack.
    // This may exceed capacity, which indicates that the caller attempted
    // to push back more values than there was space to store.
    unsigned pushes() const {
        return storage_->size;
    }

    bool overflow() const {
        return storage_->size>capacity();
    }

    // The number of values stored in the stack.
    unsigned size() const {
        return std::min(storage_->size, storage_->capacity);
    }

    // The maximum number of items that can be stored in the stack.
    unsigned capacity() const {
        return storage_->capacity;
    }

    storage_type& storage() {
        return *storage_;
    }

    value_type& operator[](unsigned i) {
        EXPECTS(i<size());
        return storage_->data[i];
    }

    value_type& operator[](unsigned i) const {
        EXPECTS(i<size());
        return storage_->data[i];
    }

    value_type* begin() {
        return storage_->data;
    }
    const value_type* begin() const {
        return storage_->data;
    }

    value_type* end() {
        // Take care of the case where size>capacity.
        return storage_->data + size();
    }
    const value_type* end() const {
        // Take care of the case where size>capacity.
        return storage_->data + size();
    }
};

} // namespace gpu
} // namespace mc
} // namespace nest
