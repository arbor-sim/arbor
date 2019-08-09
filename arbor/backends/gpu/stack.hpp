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

private:
    // pointer in GPU memory
    storage_type* device_storage_;

    // copy of the device_storage in host
    storage_type host_storage_;
    gpu_context_handle gpu_context_;

    // copy of data from GPU memory, to be manually refreshed before access
    std::vector<T> data_;

    void create_storage(unsigned n) {
        data_.reserve(n);

        host_storage_.capacity = n;
        host_storage_.stores = 0u;
        host_storage_.data = n>0u ? allocator<value_type>().allocate(n): nullptr;

        device_storage_ = allocator<storage_type>().allocate(1);
        memory::cuda_memcpy_h2d(device_storage_, &host_storage_, sizeof(storage_type));
    }

public:

    stack& operator=(const stack& other) = delete;
    stack(const stack& other) = delete;
    stack() = delete;

    stack(gpu_context_handle h): gpu_context_(h) {
        host_storage_.data = nullptr;
        device_storage_ = nullptr;
    }

    stack& operator=(stack&& other) {
        gpu_context_ = other.gpu_context_;
        std::swap(device_storage_, other.device_storage_);
        std::swap(host_storage_, other.host_storage_);
        std::swap(data_, other.data_);
        return *this;
    }

    stack(stack&& other) {
        *this = std::move(other);
    }

    explicit stack(unsigned capacity, const gpu_context_handle& gpu_ctx): gpu_context_(gpu_ctx) {
        create_storage(capacity);
    }

    ~stack() {
        if (host_storage_.data) {
            allocator<value_type>().deallocate(host_storage_.data, host_storage_.capacity);
        }
        allocator<storage_type>().deallocate(device_storage_, sizeof(storage_type));
    }

    // After this call both host and device storage are synchronized to the GPU
    // state before the call.
    void update_host() {
        memory::cuda_memcpy_d2h(&host_storage_, device_storage_, sizeof(storage_type));

        auto num = size();
        data_.resize(num);
        auto bytes = num*sizeof(T);
        memory::cuda_memcpy_d2h(data_.data(), host_storage_.data, bytes);
    }

    // After this call both host and device storage are synchronized to empty state.
    void clear() {
        host_storage_.stores = 0u;
        memory::cuda_memcpy_h2d(device_storage_, &host_storage_, sizeof(storage_type));
        data_.clear();
    }

    // The information returned by the calls below may be out of sync with the
    // version on the GPU if the GPU storage has been modified since the last
    // call to update_host().
    storage_type get_storage_copy() const {
        return host_storage_;
    }

    const std::vector<value_type>& data() const {
        return data_;
    }

    // The number of items that have been pushed back on the stack.
    // This may exceed capacity, which indicates that the caller attempted
    // to push back more values than there was space to store.
    unsigned pushes() const {
        return host_storage_.stores;
    }

    bool overflow() const {
        return host_storage_.stores>host_storage_.capacity;
    }

    // The number of values stored in the stack.
    unsigned size() const {
        return std::min(host_storage_.stores, host_storage_.capacity);
    }

    // The maximum number of items that can be stored in the stack.
    unsigned capacity() const {
        return host_storage_.capacity;
    }

    // This returns a non-const reference to the unerlying device storage so
    // that it can be passed to GPU kernels that need to modify the stack.
    storage_type& storage() {
        return *device_storage_;
    }

    const value_type& operator[](unsigned i) const {
        arb_assert(i<size());
        return data_[i];
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
