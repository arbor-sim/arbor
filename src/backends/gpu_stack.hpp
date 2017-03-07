#pragma once

#include <memory/allocator.hpp>

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
class gpu_stack {
    using value_type = T;
    using allocator = memory::managed_allocator<value_type>;

    // The number of items of type value_type that can be stored in the stack
    unsigned capacity_;

    // The number of items that have been stored
    unsigned size_;

    // Memory containing the value buffer
    // Stored in managed memory to facilitate host-side access of values
    // pushed from kernels on the device.
    value_type* data_;

public:

    gpu_stack(unsigned capacity):
        capacity_(capacity), size_(0u)
    {
        data_ = allocator().allocate(capacity_);
    }

    ~gpu_stack() {
        allocator().deallocate(data_, capacity_);
    }

    // Append a new value to the stack.
    // The value will only be appended if do_push is true.
    __device__
    void push_back(const value_type& value) {
        // Atomically increment the size_ counter. The atomicAdd returns
        // the value of size_ before the increment, which is the location
        // at which this thread can store value.
        unsigned position = atomicAdd(&size_, 1u);

        // It is possible that size_>capacity_. In this case, only capacity_
        // entries are stored, and additional values are lost. The size_
        // will contain the total number of attempts to push,
        if (position<capacity_) {
            data_[position] = value;
        }
    }

    __host__
    void clear() {
        size_ = 0;
    }

    // The number of items that have been pushed back on the stack.
    // size may exceed capacity, which indicates that the caller attempted
    // to push back more values than there was space to store.
    __host__ __device__
    unsigned size() const {
        return size_;
    }

    // The maximum number of items that can be stored in the stack.
    __host__ __device__
    unsigned capacity() const {
        return capacity_;
    }

    value_type& operator[](unsigned i) {
        EXPECTS(i<size_ && i<capacity_);
        return data_[i];
    }

    value_type& operator[](unsigned i) const {
        EXPECTS(i<size_ && i<capacity_);
        return data_[i];
    }

    value_type* begin() {
        return data_;
    }
    const value_type* begin() const {
        return data_;
    }

    value_type* end() {
        // Take care of the case where size_>capacity_.
        return data_ + (size_>capacity_? capacity_: size_);
    }
    const value_type* end() const {
        // Take care of the case where size_>capacity_.
        return data_ + (size_>capacity_? capacity_: size_);
    }
};

} // namespace gpu
} // namespace mc
} // namespace nest
