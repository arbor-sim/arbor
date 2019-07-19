#pragma once

#include <algorithm>
#include <memory>

#include <arbor/assert.hpp>

#include "gpu_context.hpp"
#include "memory/allocator.hpp"
#include "memory/cuda_wrappers.hpp"
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
    using allocator = memory::cuda_allocator<U>;

    using storage_type = stack_storage<value_type>;

    using gpu_context_handle = std::shared_ptr<arb::gpu_context>;

protected:
    // pointer in GPU memory
    storage_type* storage_;
    storage_type host_copy_;
    gpu_context_handle gpu_context_;

    storage_type* create_storage(unsigned n) {
        storage_type p;
        p.capacity = n;
        p.stores = 0u;
        p.data = n>0u ? allocator<value_type>().allocate(n): nullptr;

        auto p_gpu = allocator<storage_type>().allocate(1);

        memory::cuda_memcpy_h2d(p_gpu, &p, sizeof(storage_type));

        host_copy_ = p;
        
        return p_gpu;
    }

public:
    // copy of data from GPU memory, to be manually refreshed before access
    std::vector<T> data_;

    stack& operator=(const stack& other) = delete;
    stack(const stack& other) = delete;

    stack(const gpu_context_handle& gpu_ctx):
        storage_(create_storage(0)), gpu_context_(gpu_ctx) {}

    stack(stack&& other): storage_(create_storage(0)), gpu_context_(other.gpu_context_) {
        std::swap(storage_, other.storage_);
    }

    stack& operator=(stack&& other) {
        std::swap(storage_, other.storage_);
        return *this;
    }

    explicit stack(unsigned capacity, const gpu_context_handle& gpu_ctx):
        storage_(create_storage(capacity)), gpu_context_(gpu_ctx) {}

    ~stack() {
        memory::cuda_sync();
        auto st = get_storage_copy();
        if (st.data) {
            allocator<value_type>().deallocate(st.data, st.capacity);
        }
    }

    std::vector<T> get_data() const {
        if (capacity() != 0u) {

            auto num = std::min(host_copy_.stores, host_copy_.capacity);
            std::vector<T> buf(host_copy_.stores);
            auto bytes = num*sizeof(T);
            memory::cuda_memcpy_d2h(buf.data(), host_copy_.data, bytes);
            
            return buf;
        }
        else {
            return std::vector<T> {};
        }

    }

    storage_type get_storage_copy() const {
        storage_type st;
        memory::cuda_memcpy_d2h(&st, storage_, sizeof(storage_type));

        return st;
    }


    void refresh_host_copy() {
        host_copy_ = get_storage_copy();
        data_ = get_data();
    }

    void clear() {
        host_copy_.stores = 0u;
        memory::cuda_memcpy_h2d(storage_, &host_copy_, sizeof(storage_type));
    }

    // The number of items that have been pushed back on the stack.
    // This may exceed capacity, which indicates that the caller attempted
    // to push back more values than there was space to store.
    unsigned pushes() const {
        return host_copy_.stores;
    }

    bool overflow() const {
        return host_copy_.stores>host_copy_.capacity;
    }

    // The number of values stored in the stack.
    unsigned size() const {
        storage_type storage_copy = get_storage_copy();
        return std::min(storage_copy.stores, storage_copy.capacity);
    }

    // The maximum number of items that can be stored in the stack.
    unsigned capacity() const {
        return host_copy_.capacity;
    }

    storage_type& storage() {
        return *storage_;
    }

    value_type& operator[](unsigned i) {
        arb_assert(i<size());
        return data_[i];
    }

    value_type& operator[](unsigned i) const {
        arb_assert(i<size());
        return data_[i];
    }

    value_type* begin() {
        return data_.data();
    }

    value_type* end() {
        // Take care of the case where size>capacity.
        return data_.data() + size();
    }

    const value_type* begin() const {
        return data_.data();
    }

    const value_type* end() const {
        // Take care of the case where size>capacity.
        return data_.data() + size();
    }

};

} // namespace gpu
} // namespace arb
