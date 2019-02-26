#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory/allocator.hpp>

namespace arb {
namespace gpu {

// used to indicate that the type pointed to by the managed_ptr is to be
// constructed in the managed_ptr constructor
struct construct_in_place_tag {};

// Like std::unique_ptr, but for CUDA managed memory.
// Handles memory allocation and freeing, and the construction and destruction
// of the type being stored in the allocated memory.
// Implemented as a stand alone type instead of as a std::unique_ptr with
// custom desctructor so that __device__ annotation can be added to members
// like ::get, ::operator*, etc., which enables the use of the smart pointer
// in device side code.
//
// It is very strongly recommended that the helper make_managed_ptr be used
// instead of directly constructing the managed_ptr.
template <typename T>
class managed_ptr {
public:

    using element_type = T;
    using pointer = element_type*;
    using reference = element_type&;
    
    managed_ptr(const managed_ptr& other) = delete;

    // Allocate memory and construct in place using args.
    // This is an extension over the std::unique_ptr interface, because the
    // point of the wrapper is to hide the complexity of allocating managed
    // memory and constructing a type in place.
    template <typename... Args>
    managed_ptr(construct_in_place_tag, Args&&... args) {
        memory::managed_allocator<element_type> allocator;
        data_ = allocator.allocate(1u);
        synchronize();
        data_ = new (data_) element_type(std::forward<Args>(args)...);
    }

    managed_ptr(managed_ptr&& other) {
        std::swap(other.data_, data_);
    }

    // pointer to the managed object
    __host__ __device__
    pointer get() const {
        return data_;
    }

    // return a reference to the managed object
    __host__ __device__
    reference operator *() const {
        return *data_;
    }

    // return a reference to the managed object
    __host__ __device__
    pointer operator->() const {
        return get();
    }

    managed_ptr& operator=(managed_ptr&& other) {
        swap(std::move(other));
        return *this;
    }

    ~managed_ptr() {
        if (is_allocated()) {
            memory::managed_allocator<element_type> allocator;
            synchronize(); // required to ensure that memory is not in use on GPU
            data_->~element_type();
            allocator.deallocate(data_, 1u);
        }
    }

    void swap(managed_ptr&& other) {
        std::swap(other.data_, data_);
    }

    __host__ __device__
    operator bool() const {
        return is_allocated();
    }

    void synchronize() const {
        cudaDeviceSynchronize();
    }

private:
    __host__ __device__
    bool is_allocated() const {
        return data_!=nullptr;
    }

    pointer data_ = nullptr;
};

// The correct way to construct a type in managed memory.
// Equivalent to std::make_unique_ptr for std::unique_ptr
template <typename T, typename... Args>
managed_ptr<T> make_managed_ptr(Args&&... args) {
    return managed_ptr<T>(construct_in_place_tag(), std::forward<Args>(args)...);
}

} // namespace gpu
} // namespace arb

