#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>

#include "Allocator.hpp"
#include "Array.hpp"
#include "definitions.hpp"
#include "Event.hpp"
#include "CudaEvent.hpp"
#include "gpu.hpp"

namespace memory {

// forward declare
template <typename T, class Allocator>
class DeviceCoordinator;

template <typename T, class Allocator>
class HostCoordinator;

namespace util {
    template <typename T, typename Allocator>
    struct type_printer<DeviceCoordinator<T,Allocator>>{
        static std::string print() {
            #if VERBOSE > 1
            return util::white("DeviceCoordinator") + "<"
                + type_printer<T>::print()
                + ", " + type_printer<Allocator>::print() + ">";
            #else
            return util::white("DeviceCoordinator")
                + "<" + type_printer<T>::print() + ">";
            #endif
        }
    };

    template <typename T, typename Allocator>
    struct pretty_printer<DeviceCoordinator<T,Allocator>>{
        static std::string print(const DeviceCoordinator<T,Allocator>& val) {
            return type_printer<DeviceCoordinator<T,Allocator>>::print();
        }
    };
} // namespace util

namespace gpu {
    // brief:
    // We have to perform some type punning to pass arbitrary POD types to the
    // GPU backend without polluting the library front end with CUDA kernels
    // that would require compilation with nvcc.
    //
    // detail:
    // The implementation takes advantage of 4 fill functions that fill GPU
    // memory with a {8, 16, 32, 64} bit unsigned integer. We want to use these
    // functions to fill a block of GPU memory with _any_ 8, 16, 32 or 64 bit POD
    // value. The technique to do this with a 64-bit double, is to first convert
    // the double into a 64-bit unsigned integer (with the same bits, not the
    // same value), then call the 64-bit fill kernel precompiled using nvcc in
    // the gpu library. This technique of converting from one type to another
    // is called type-punning. There are plenty of subtle problems with this, due
    // to C++'s strict aliasing rules, that require memcpy of single bytes if
    // alignment of the two types does not match.

    #define FILL(N) \
    template <typename T> \
    typename std::enable_if<sizeof(T)==sizeof(uint ## N ## _t)>::type \
    fill(T* ptr, T value, size_t n) { \
        using I = uint ## N ## _t; \
        I v; \
        if(alignof(T)==alignof(I)) { \
            *reinterpret_cast<T*>(&v) = value; \
        } \
        else { \
            std::copy_n( \
                reinterpret_cast<char*>(&value), \
                sizeof(T), \
                reinterpret_cast<char*>(&v) \
            ); \
        } \
        fill ## N(reinterpret_cast<I*>(ptr), v, n); \
    }

    FILL(8)
    FILL(16)
    FILL(32)
    FILL(64)
}

template <typename T>
class ConstDeviceReference {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    ConstDeviceReference(const_pointer p) : pointer_(p) {}

    operator T() const {
        T tmp;
        auto success
            = cudaMemcpy(&tmp, pointer_, sizeof(T), cudaMemcpyDeviceToHost);
        if(success != cudaSuccess) {
            std::cerr << util::red("error")
                      << " bad CUDA memcopy, unable to copy " << sizeof(T)
                      << " bytes from host to device";
            exit(-1);
        }
        return T(tmp);
    }

protected:
    template <typename Other>
    void operator =(Other&&) {}

    const_pointer pointer_;
};

template <typename T>
class DeviceReference {
public:
    using value_type = T;
    using pointer = value_type*;

    DeviceReference(pointer p) : pointer_(p) {}

    DeviceReference& operator = (const T& value) {
        auto success =
            cudaMemcpy(pointer_, &value, sizeof(T), cudaMemcpyHostToDevice);
        if(success != cudaSuccess) {
            std::cerr << util::red("error")
                      << " bad CUDA memcopy, unable to copy " << sizeof(T)
                      << " bytes from host to device";
            exit(-1);
        }
        return *this;
    }

    operator T() const {
        T tmp;
        auto success =
            cudaMemcpy(&tmp, pointer_, sizeof(T), cudaMemcpyDeviceToHost);
        if(success != cudaSuccess) {
            std::cerr << util::red("error")
                      << " bad CUDA memcopy, unable to copy " << sizeof(T)
                      << " bytes from device to host";
            exit(-1);
        }
        return T(tmp);
    }

private:
    pointer pointer_;
};


template <typename T, class Allocator_=CudaAllocator<T> >
class DeviceCoordinator {
public:
    using value_type = T;
    using Allocator = typename Allocator_::template rebind<value_type>;

    using pointer       = value_type*;
    using const_pointer = const value_type*;
    using reference       = DeviceReference<value_type>;
    using const_reference = ConstDeviceReference<value_type>;

    using view_type = ArrayView<value_type, DeviceCoordinator>;

    using size_type       = types::size_type;
    using difference_type = types::difference_type;

    template <typename Tother>
    using rebind = DeviceCoordinator<Tother, Allocator>;

    view_type allocate(size_type n) {
        Allocator allocator;

        pointer ptr = n>0 ? allocator.allocate(n) : nullptr;

        #ifdef VERBOSE
        std::cerr << util::type_printer<DeviceCoordinator>::print()
                  << util::blue("::allocate") << "(" << n << ")"
                  << (ptr==nullptr && n>0 ? " failure" : " success")
                  << std::endl;
        #endif

        return view_type(ptr, n);
    }

    void free(view_type& rng) {
        Allocator allocator;

        if(rng.data())
            allocator.deallocate(rng.data(), rng.size());

        #ifdef VERBOSE
        std::cerr << util::type_printer<DeviceCoordinator>::print()
                  << "::free()" << std::endl;
        #endif

        impl::reset(rng);
    }

    // copy memory from one gpu range to another
    void copy(const view_type &from, view_type &to) {
        assert(from.size()==to.size());
        assert(!from.overlaps(to));

        auto status = cudaMemcpy(
                reinterpret_cast<void*>(to.begin()),
                reinterpret_cast<const void*>(from.begin()),
                from.size()*sizeof(value_type),
                cudaMemcpyDeviceToDevice
        );
        if(status != cudaSuccess) {
            std::cerr << util::red("error") << " bad CUDA memcopy, unable to copy " << sizeof(T)*from.size() << " bytes from device to device";
            exit(-1);
        }
    }

    // generates compile time error if there is an attempt to copy from memory
    // that is managed by a coordinator for which there is no specialization
    template <class CoordOther>
    void copy( const ArrayView<value_type, CoordOther> &from,
               view_type &to) {
        static_assert(true, "DeviceCoordinator: unable to copy from other Coordinator");
    }

    template <class Alloc>
    std::pair<SynchEvent, view_type>
    copy(const ArrayView<value_type, HostCoordinator<value_type, Alloc>> &from,
         view_type &to) {
        assert(from.size()==to.size());

        #ifdef VERBOSE
        using oType = ArrayView<value_type, HostCoordinator<value_type, Alloc>>;
        std::cout << util::pretty_printer<DeviceCoordinator>::print(*this)
                  << "::" << util::blue("copy") << "(asynchronous, " << from.size() << ")"
                  << "\n  " << util::type_printer<oType>::print() << " @ " << from.data()
                  << util::yellow(" -> ")
                  << util::type_printer<view_type>::print() << " @ " << to.data() << std::endl;
        #endif

        auto status = cudaMemcpy(
                reinterpret_cast<void*>(to.begin()),
                reinterpret_cast<const void*>(from.begin()),
                from.size()*sizeof(value_type),
                cudaMemcpyHostToDevice
        );
        if(status != cudaSuccess) {
            std::cerr << util::red("error") << " bad CUDA memcopy, unable to copy " << sizeof(T)*from.size() << " bytes from host to device";
            exit(-1);
        }

        return std::make_pair(SynchEvent(), to);
    }

    template <size_t alignment>
    std::pair<CudaEvent, view_type>
    copy(const ArrayView<
                value_type,
                HostCoordinator<
                    value_type,
                    PinnedAllocator<
                        value_type,
                        alignment>>> &from,
         view_type &to) {
        assert(from.size()==to.size());

        #ifdef VERBOSE
        using oType = ArrayView< value_type, HostCoordinator< value_type, PinnedAllocator< value_type, alignment>>>;
        std::cout << util::pretty_printer<DeviceCoordinator>::print(*this)
                  << "::" << util::blue("copy") << "(asynchronous, " << from.size() << ")"
                  << "\n  " << util::type_printer<oType>::print() << " @ " << from.data()
                  << util::yellow(" -> ")
                  << util::type_printer<view_type>::print() << " @ " << to.data() << std::endl;
        #endif

        auto status = cudaMemcpy(
                reinterpret_cast<void*>(to.begin()),
                reinterpret_cast<const void*>(from.begin()),
                from.size()*sizeof(value_type),
                cudaMemcpyHostToDevice
        );
        if(status != cudaSuccess) {
            std::cerr << util::red("error") << " bad CUDA memcopy, unable to copy " << sizeof(T)*from.size() << " bytes from host to device";
            exit(-1);
        }

        CudaEvent event;
        return std::make_pair(event, to);
    }

    // fill memory
    void set(view_type &rng, value_type value) {
        gpu::fill<value_type>(rng.data(), value, rng.size());
    }

    // generate reference objects for a raw pointer.
    reference make_reference(value_type* p) {
        return reference(p);
    }

    const_reference make_reference(value_type const* p) const {
        return const_reference(p);
    }

    static constexpr
    auto alignment() -> decltype(Allocator_::alignment()) {
        return Allocator_::alignment();
    }

    static constexpr
    bool is_malloc_compatible() {
        return Allocator_::is_malloc_compatible();
    }
};

} // namespace memory

