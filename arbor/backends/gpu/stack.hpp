#pragma once

#include <algorithm>

#include <arbor/assert.hpp>

#include "backends/gpu/managed_ptr.hpp"
#include "memory/allocator.hpp"
#include "stack_storage.hpp"

namespace arb {
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
    managed_ptr<storage_type> storage_;

    managed_ptr<storage_type> create_storage(unsigned n, unsigned cuda_arch) {
        auto p = make_managed_ptr<storage_type>(cuda_arch);
        p->capacity = n;
        p->stores = 0;
        p->data = n? allocator<value_type>().allocate(n): nullptr;
        return p;
    }

public:
    stack& operator=(const stack& other) = delete;
    stack(const stack& other) = delete;

    stack(unsigned cuda_arch): storage_(create_storage(0, cuda_arch)) {}

    stack(stack&& other): storage_(create_storage(0)) {
        std::swap(storage_, other.storage_);
    }

    stack& operator=(stack&& other) {
        std::swap(storage_, other.storage_);
        return *this;
    }

    explicit stack(unsigned capacity, unsigned cuda_arch): storage_(create_storage(capacity, cuda_arch)) {}

    ~stack() {
        storage_.synchronize();
        if (storage_->data) {
            allocator<value_type>().deallocate(storage_->data, storage_->capacity);
        }
    }

    // Perform any required synchronization if concurrent host-side access is not supported.
    // (Correctness still requires that GPU operations on this stack are complete.)
    void host_access() const {
        storage_.host_access();
    }

    void clear() {
        storage_->stores = 0u;
    }

    // The number of items that have been pushed back on the stack.
    // This may exceed capacity, which indicates that the caller attempted
    // to push back more values than there was space to store.
    unsigned pushes() const {
        return storage_->stores;
    }

    bool overflow() const {
        return storage_->stores>capacity();
    }

    // The number of values stored in the stack.
    unsigned size() const {
        return std::min(storage_->stores, storage_->capacity);
    }

    // The maximum number of items that can be stored in the stack.
    unsigned capacity() const {
        return storage_->capacity;
    }

    storage_type& storage() {
        return *storage_;
    }

    value_type& operator[](unsigned i) {
        arb_assert(i<size());
        return storage_->data[i];
    }

    value_type& operator[](unsigned i) const {
        arb_assert(i<size());
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
} // namespace arb
