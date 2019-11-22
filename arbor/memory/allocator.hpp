#pragma once

#include <limits>

#include "cuda_wrappers.hpp"
#include "definitions.hpp"
#include "util.hpp"

namespace arb {
namespace memory {

namespace impl {
    using size_type = std::size_t;

    constexpr bool
    is_power_of_two(size_type x) {
        return !(x&(x-1));
    }

    // returns the smallest power of two that is strictly greater than x
    constexpr size_type
    next_power_of_two(size_type x, size_type p) {
        return x==0 ? p : next_power_of_two(x-(x&p), p<<1);
    }

    // returns the smallest power of two that is greater than or equal to x
    constexpr size_type
    round_up_power_of_two(size_type x) {
        return is_power_of_two(x) ? x : next_power_of_two(x, 1);
    }

    // returns the smallest power of two that is greater than
    // or equal to sizeof(T), and greater than or equal to sizeof(void*)
    template <typename T>
    constexpr size_type
    minimum_possible_alignment() {
        return round_up_power_of_two(sizeof(T)) < sizeof(void*)
                    ?   sizeof(void*)
                    :   round_up_power_of_two(sizeof(T));
    }

    // Calculate the padding that has to be added to an array of T of length n
    // so that the size of the array in bytes is a multiple of sizeof(T).
    // The returned value is in terms of T, not bytes.
    template<typename T>
    size_type
    get_padding(const size_type alignment, size_type n) {
        // calculate the remaninder in bytes for n items of size sizeof(T)
        auto remainder = (n*sizeof(T)) % alignment;

        // calculate padding in bytes
        return remainder ? (alignment - remainder)/sizeof(T) : 0;
    }

    // allocate memory with alignment specified as a template parameter
    // returns nullptr on failure
    template <typename T, size_type alignment=minimum_possible_alignment<T>()>
    T* aligned_malloc(size_type size) {
        // double check that alignment is a multiple of sizeof(void*),
        // which is a prerequisite for posix_memalign()
        static_assert( !(alignment%sizeof(void*)),
                "alignment is not a multiple of sizeof(void*)");
        static_assert( is_power_of_two(alignment),
                "alignment is not a power of two");
        void *ptr;
        int result = posix_memalign(&ptr, alignment, size*sizeof(T));
        if(result) {
            return nullptr;
        }
        return reinterpret_cast<T*>(ptr);
    }

    template <size_type Alignment>
    class aligned_policy {
    public:
        void *allocate_policy(size_type size) {
            return reinterpret_cast<void *>(aligned_malloc<char, Alignment>(size));
        }

        void free_policy(void *ptr) {
            free(ptr);
        }

        static constexpr size_type alignment() {
            return Alignment;
        }
        static constexpr bool is_malloc_compatible() {
            return true;
        }
    };

    namespace cuda {
        template <size_type Alignment>
        class pinned_policy {
        public:
            void *allocate_policy(size_type size) {
                // first allocate memory with the desired alignment
                void* ptr = reinterpret_cast<void *>
                                (aligned_malloc<char, Alignment>(size));

                if (!ptr) {
                    return nullptr;
                }

                // register the memory with CUDA
                if (!cuda_host_register(ptr, size)) {
                    free(ptr);
                    ptr = nullptr;
                }

                return ptr;
            }

            void free_policy(void *ptr) {
                if (!ptr) {
                    return;
                }
                cuda_host_unregister(ptr);
                free(ptr);
            }

            static constexpr size_type alignment() {
                return Alignment;
            }
            static constexpr bool is_malloc_compatible() {
                return true;
            }
        };

        class device_policy {
        public:
            void *allocate_policy(size_type size) {
                return cuda_malloc(size);
            }

            void free_policy(void *ptr) {
                cuda_free(ptr);
            }

            // memory allocated using cudaMalloc has alignment of 256 bytes
            static constexpr size_type alignment() {
                return 256;
            }
            static constexpr bool is_malloc_compatible() {
                return true;
            }
        };
    } // namespace cuda
} // namespace impl

template<typename T, typename Policy >
class allocator : public Policy {
    using Policy::allocate_policy;
    using Policy::free_policy;
public:
    using Policy::alignment;

    using value_type    = T;
    using pointer       = value_type*;
    using const_pointer = const value_type*;
    using reference     = value_type&;
    using const_reference = const value_type& ;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

public:
    template <typename U>
    using rebind = allocator<U, Policy>;

public:
    allocator() {}
    ~allocator() {}
    allocator(allocator const&) {}

    pointer address(reference r) {
        return &r;
    }
    const_pointer address(const_reference r) {
        return &r;
    }

    pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = 0) {
        if (cnt) {
            return reinterpret_cast<T*>(allocate_policy(cnt*sizeof(T)));
        }
        return nullptr;
    }

    void deallocate(pointer p, size_type cnt) {
        if (p) {
            free_policy(p);
        }
    }

    size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    void construct(pointer p, const T& t) {
        new(p) T(t);
    }

    void destroy(pointer p) {
        p->~T();
    }

    bool operator==(allocator const&) {
        return true;
    }
    bool operator!=(allocator const& a) {
        return !operator==(a);
    }
};

// pretty printers
namespace util {
    template <size_t Alignment>
    struct type_printer<impl::aligned_policy<Alignment>>{
        static std::string print() {
            std::stringstream str;
            str << "aligned_policy<" << Alignment << ">";
            return str.str();
        }
    };

    template <size_t Alignment>
    struct type_printer<impl::cuda::pinned_policy<Alignment>>{
        static std::string print() {
            std::stringstream str;
            str << "pinned_policy<" << Alignment << ">";
            return str.str();
        }
    };

    template <>
    struct type_printer<impl::cuda::device_policy>{
        static std::string print() {
            return std::string("device_policy");
        }
    };

    template <typename T, typename Policy>
    struct type_printer<allocator<T,Policy>>{
        static std::string print() {
            std::stringstream str;
            str << util::white("allocator") << "<" << type_printer<T>::print()
                << ", " << type_printer<Policy>::print() << ">";
            return str.str();
        }
    };
} // namespace util

// helper for generating an aligned allocator
template <class T, size_t alignment=impl::minimum_possible_alignment<T>()>
using aligned_allocator = allocator<T, impl::aligned_policy<alignment>>;

// For pinned and allocation set the default alignment to correspond to
// the alignment of 1024 bytes, because pinned memory is allocated at
// page boundaries. It is allocated at page boundaries (typically 4k),
// however in practice it will return pointers that are 1k aligned.
template <class T, size_t alignment=1024>
using pinned_allocator = allocator<T, impl::cuda::pinned_policy<alignment>>;

// use 256 as default alignment, because that is the default for cudaMalloc
template <class T, size_t alignment=256>
using cuda_allocator = allocator<T, impl::cuda::device_policy>;

} // namespace memory
} // namespace arb
