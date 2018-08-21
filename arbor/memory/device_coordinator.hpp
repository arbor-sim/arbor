#pragma once

#include <cstdint>
#include <exception>

#include <arbor/assert.hpp>

#include "allocator.hpp"
#include "array.hpp"
#include "cuda_wrappers.hpp"
#include "definitions.hpp"
#include "fill.hpp"
#include "util.hpp"

namespace arb {
namespace memory {

// forward declare
template <typename T, class Allocator>
class device_coordinator;

template <typename T, class Allocator>
class host_coordinator;

namespace util {
    template <typename T, typename Allocator>
    struct type_printer<device_coordinator<T,Allocator>>{
        static std::string print() {
            #if VERBOSE > 1
            return util::white("device_coordinator") + "<"
                + type_printer<T>::print()
                + ", " + type_printer<Allocator>::print() + ">";
            #else
            return util::white("device_coordinator")
                + "<" + type_printer<T>::print() + ">";
            #endif
        }
    };

    template <typename T, typename Allocator>
    struct pretty_printer<device_coordinator<T,Allocator>>{
        static std::string print(const device_coordinator<T,Allocator>& val) {
            return type_printer<device_coordinator<T,Allocator>>::print();
        }
    };
} // namespace util

template <typename T>
class const_device_reference {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    const_device_reference(const_pointer p) : pointer_(p) {}

    operator T() const {
        T tmp;
        cuda_memcpy_d2h(&tmp, pointer_, sizeof(T));
        return tmp;
    }

protected:
    template <typename Other>
    void operator =(Other&&) {}

    const_pointer pointer_;
};

template <typename T>
class device_reference {
public:
    using value_type = T;
    using pointer = value_type*;

    device_reference(pointer p) : pointer_(p) {}

    device_reference& operator=(const T& value) {
        cuda_memcpy_h2d(pointer_, &value, sizeof(T));
        return *this;
    }

    device_reference& operator=(const device_reference& ref) {
        cuda_memcpy_d2d(pointer_, ref.pointer_, sizeof(T));
    }

    operator T() const {
        T tmp;
        cuda_memcpy_d2h(&tmp, pointer_, sizeof(T));
        return tmp;
    }

private:
    pointer pointer_;
};

template <typename T, class Allocator_= cuda_allocator<T> >
class device_coordinator {
public:
    using value_type = T;
    using Allocator = typename Allocator_::template rebind<value_type>;

    using pointer       = value_type*;
    using const_pointer = const value_type*;
    using reference       = device_reference<value_type>;
    using const_reference = const_device_reference<value_type>;

    using view_type = array_view<value_type, device_coordinator>;
    using const_view_type = const_array_view<value_type, device_coordinator>;

    using size_type       = types::size_type;
    using difference_type = types::difference_type;

    template <typename Tother>
    using rebind = device_coordinator<Tother, Allocator>;

    view_type allocate(size_type n) {
        Allocator allocator;

        pointer ptr = n>0 ? allocator.allocate(n) : nullptr;

        #ifdef VERBOSE
        std::cerr << util::type_printer<device_coordinator>::print()
                  << util::blue("::allocate") << "(" << n << ") -> " << util::print_pointer(ptr)
                  << "\n";
        #endif

        return view_type(ptr, n);
    }

    void free(view_type& rng) {
        Allocator allocator;

        #ifdef VERBOSE
        std::cerr << util::type_printer<device_coordinator>::print()
                  << util::blue("::free") << "(size=" << rng.size() << ", pointer=" << util::print_pointer(rng.data()) << ")\n";
        #endif

        if(rng.data()) {
            allocator.deallocate(rng.data(), rng.size());
        }

        impl::reset(rng);
    }

    // copy memory from one gpu range to another
    void copy(const_view_type from, view_type to) {
        #ifdef VERBOSE
        std::cerr << util::type_printer<device_coordinator>::print()
                  << util::blue("::copy") << "(size=" << from.size() << ") "
                  << util::print_pointer(from.data()) << " -> "
                  << util::print_pointer(to.data()) << "\n";
        #endif
        arb_assert(from.size()==to.size());
        arb_assert(!from.overlaps(to));

        cuda_memcpy_d2d(to.data(), from.data(), from.size()*sizeof(value_type));
    }

    // copy memory from gpu to host
    template <typename Allocator>
    void copy(
        const_view_type& from,
        array_view<value_type, host_coordinator<value_type, Allocator>>& to)
    {
        #ifdef VERBOSE
        std::cerr << util::type_printer<device_coordinator>::print()
                  << util::blue("::copy") << "(d2h, size=" << from.size() << ") "
                  << util::print_pointer(from.data()) << " -> "
                  << util::print_pointer(to.data()) << "\n";
        #endif
        arb_assert(from.size()==to.size());

        cuda_memcpy_d2h(to.data(), from.data(), from.size()*sizeof(value_type));
    }

    // copy memory from host to gpu
    template <class Alloc>
    void copy(
        const_array_view<value_type, host_coordinator<value_type, Alloc>> from,
        view_type to)
    {
        #ifdef VERBOSE
        std::cerr << util::type_printer<device_coordinator>::print()
                  << util::blue("::copy") << "(size=" << from.size() << ") "
                  << util::print_pointer(from.data()) << " -> "
                  << util::print_pointer(to.data()) << "\n";
        #endif
        arb_assert(from.size()==to.size());

        cuda_memcpy_h2d(to.data(), from.data(), from.size()*sizeof(value_type));
    }

    // copy from pinned memory to device
    // TODO : asynchronous version
    template <size_t alignment>
    void copy(
        const_array_view<value_type, host_coordinator<value_type, pinned_allocator<value_type, alignment>>> from,
        view_type to)
    {
        #ifdef VERBOSE
        std::cerr << util::type_printer<device_coordinator>::print()
                  << util::blue("::copy") << "(size=" << from.size() << ") " << from.data() << " -> " << to.data() << "\n";
        #endif
        arb_assert(from.size()==to.size());

        #ifdef VERBOSE
        using oType = array_view< value_type, host_coordinator< value_type, pinned_allocator< value_type, alignment>>>;
        std::cout << util::pretty_printer<device_coordinator>::print(*this)
                  << "::" << util::blue("copy") << "(asynchronous, " << from.size() << ")"
                  << "\n  " << util::type_printer<oType>::print() << " "
                  << util::print_pointer(from.data()) << " -> "
                  << util::print_pointer(to.data()) << "\n";
        #endif

        cuda_memcpy_h2d(to.begin(), from.begin(), from.size()*sizeof(value_type));
    }

    // generates compile time error if there is an attempt to copy from memory
    // that is managed by a coordinator for which there is no specialization
    template <class CoordOther>
    void copy(const array_view<value_type, CoordOther>& from, view_type& to) {
        static_assert(true, "device_coordinator: unable to copy from other Coordinator");
    }

    // fill memory
    void set(view_type &rng, value_type value) {
        if (rng.size()) {
            arb::gpu::fill<value_type>(rng.data(), value, rng.size());
        }
    }

    // generate reference objects for a raw pointer.
    reference make_reference(value_type* p) {
        return reference(p);
    }

    const_reference make_reference(value_type const* p) const {
        return const_reference(p);
    }

    static constexpr
    auto alignment() {
        return Allocator_::alignment();
    }

    static constexpr
    bool is_malloc_compatible() {
        return Allocator_::is_malloc_compatible();
    }
};

} // namespace memory
} // namespace arb
