#pragma once
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

// after: https://github.com/KonanM/small_vector (Unlicense)
namespace arb::util {

template<typename T, size_t N = 8, typename NonReboundT = T>
struct smallvec_allocator {
    alignas(alignof(T)) std::byte buffer_[N * sizeof(T)];
    std::allocator<T> alloc_{};
    bool buffer_used_ = false;

    using value_type = T;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;
    using is_always_equal = std::false_type;

    constexpr smallvec_allocator() noexcept = default;
    template<class U> constexpr smallvec_allocator(const smallvec_allocator<U, N, NonReboundT>&) noexcept {}
    template <class U> struct rebind { typedef smallvec_allocator<U, N, NonReboundT> other; };

    constexpr smallvec_allocator(const smallvec_allocator& other) noexcept : buffer_used_(other.buffer_used_) {}
    constexpr smallvec_allocator& operator=(const smallvec_allocator& other) noexcept { buffer_used_ = other.buffer_used_; return *this; }
    constexpr smallvec_allocator(smallvec_allocator&&) noexcept {}
    constexpr smallvec_allocator& operator=(const smallvec_allocator&&) noexcept { return *this; }

    [[nodiscard]] constexpr T* allocate(const size_t n) {
        // when the allocator was rebound we don't want to use the small buffer
        if constexpr (std::is_same_v<T, NonReboundT>) {
            if (n <= N) {
                buffer_used_ = true;
                // can use small buffer return a pointer to it
                return reinterpret_cast<T*>(&buffer_);
            }
        }
        buffer_used_ = false;
        //otherwise use the default allocator
        return alloc_.allocate(n);
    }
    constexpr void deallocate(void* p, const size_t n) {
      // don't deallocate if small buffer is in use
      if (&buffer_ != p) alloc_.deallocate(static_cast<T*>(p), n);
      buffer_used_ = false;
    }

    friend constexpr bool operator==(const smallvec_allocator& lhs, const smallvec_allocator& rhs) { return !lhs.buffer_used_ && !rhs.buffer_used_; }
    friend constexpr bool operator!=(const smallvec_allocator& lhs, const smallvec_allocator& rhs) { return !(lhs == rhs); }
};

template<typename T, size_t N = 8>
struct smallvec : public std::vector<T, smallvec_allocator<T, N>>{
    using vector_type = std::vector<T, smallvec_allocator<T, N>>;
    constexpr smallvec() noexcept { vector_type::reserve(N); }
    smallvec(const smallvec&) = default;
    smallvec& operator=(const smallvec&) = default;
    smallvec(smallvec&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (other.size() <= N) vector_type::reserve(N);
        vector_type::operator=(std::move(other));
    }
    smallvec& operator=(smallvec&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (other.size() <= N) vector_type::reserve(N);
        vector_type::operator=(std::move(other));
        return *this;
    }
    // use the default constructor first to reserve then construct the values
    explicit smallvec(size_t count): smallvec() { vector_type::resize(count); }
    smallvec(size_t count, const T& value): smallvec() { vector_type::assign(count, value); }
    template<typename It> smallvec(It first, It last): smallvec() { vector_type::insert(vector_type::begin(), first, last); }
    smallvec(std::initializer_list<T> init): smallvec() { vector_type::insert(vector_type::begin(), init); }
    friend void swap(smallvec& a, smallvec& b) noexcept {
        using std::swap;
        swap(static_cast<vector_type&>(a), static_cast<vector_type&>(b));
    }
};
} // arb::util
