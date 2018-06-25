#pragma once

#include <memory>
#include <system_error>
#include <utility>

#include <iostream>

// Allocator with run-time alignment and padding guarantees.
//
// With an alignment value of `n`, any allocations will be
// aligned to have a starting address of a multiple of `n`,
// and the size of the allocation will be padded so that the
// one-past-the-end address is also a multiple of `n`.
//
// Any alignment `n` specified must be a power of two.
//
// Assignment operations propagate the alignment/padding, so that
// e.g.
// ```
//     std::vector<int, padded_allocator<int>> a(100, 32), b(50, 64);
//     a = b;
//     assert(a.get_allocator().alignment()==64);
// ```
// will pass, and the vector `a` will require reallocation.
// Correspondingly, we have to return `false`
// for the allocator equality test if the alignments differ.

namespace arb {
namespace util {

template <typename T = void>
struct padded_allocator {
    using value_type = T;
    using pointer = T*;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    padded_allocator() noexcept {}

    template <typename U>
    padded_allocator(const padded_allocator<U>& b) noexcept: alignment_(b.alignment()) {}

    explicit padded_allocator(std::size_t alignment): alignment_(alignment) {
        if (!alignment_ || (alignment_&(alignment_-1))) {
            throw std::range_error("alignment must be positive power of two");
        }
    }

    padded_allocator select_on_container_copy_construction() const noexcept {
        return *this;
    }

    pointer allocate(std::size_t n) {
        if (n>std::size_t(-1)/sizeof(T)) {
            throw std::bad_alloc();
        }

        void* mem = nullptr;
        std::size_t size = round_up(n*sizeof(T), alignment_);
        std::size_t pm_align = std::max(alignment_, sizeof(void*));

        if (auto err = posix_memalign(&mem, pm_align, size)) {
            throw std::system_error(err, std::generic_category(), "posix_memalign");
        }
        return static_cast<pointer>(mem);
    }

    void deallocate(pointer p, std::size_t n) {
        std::free(p);
    }

    bool operator==(const padded_allocator& a) const { return alignment_==a.alignment_; }
    bool operator!=(const padded_allocator& a) const { return !(*this==a); }

    std::size_t alignment() const { return alignment_; }

private:
    // Start address and one-past-the-end address a multiple of alignment:
    std::size_t alignment_ = 1;

    static std::size_t round_up(std::size_t v, std::size_t b) {
        std::size_t m = v%b;
        return v-m+(m? b: 0);
    }
};

} // namespace util
} // namespace arb
