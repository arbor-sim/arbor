#pragma once

#include <algorithm>
#include <memory>
#include <string>

#include <arbor/assert.hpp>

#include "gpu_wrappers.hpp"
#include "definitions.hpp"
#include "array.hpp"
#include "allocator.hpp"
#include "util.hpp"

namespace arb {
namespace memory {

// forward declare for type printers
template <typename T, class Allocator>
class host_coordinator;

template <typename T, class Allocator>
class device_coordinator;

namespace util {
    template <typename T, typename Allocator>
    struct type_printer<host_coordinator<T,Allocator>>{
        static std::string print() {
            #if VERBOSE>1
            return util::white("host_coordinator") + "<" + type_printer<T>::print()
                   + ", " + type_printer<Allocator>::print() + ">";
            #else
            return util::white("host_coordinator") + "<" + type_printer<T>::print() + ">";
            #endif
        }
    };

    template <typename T, typename Allocator>
    struct pretty_printer<host_coordinator<T,Allocator>>{
        static std::string print(const host_coordinator<T,Allocator>& val) {
            return type_printer<host_coordinator<T,Allocator>>::print();;
        }
    };
} // namespace util

template <typename T, class Allocator=aligned_allocator<T> >
class host_coordinator {
public:
    using value_type = T;

    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using view_type       = array_view<value_type, host_coordinator>;
    using const_view_type = const_array_view<value_type, host_coordinator>;
    using size_type       = types::size_type;
    using difference_type = types::difference_type;

    // rebind host_coordinator with another type
    template <typename Tother>
    using rebind = host_coordinator<Tother, Allocator>;

    view_type allocate(size_type n) {
        typename Allocator::template rebind<value_type> allocator;

        pointer ptr = n>0 ? allocator.allocate(n) : nullptr;

        #ifdef VERBOSE
        std::cerr << util::type_printer<host_coordinator>::print()
                  << "::" + util::blue("alocate") << "(" << n
                  << " [" << n*sizeof(value_type) << " bytes]) @ " << ptr
                  << std::endl;
        #endif

        return view_type(ptr, n);
    }

    void free(view_type& rng) {
        typename Allocator::template rebind<value_type> allocator;

        if(rng.data()) {
        #ifdef VERBOSE
            std::cerr << util::type_printer<host_coordinator>::print()
                      << "::" + util::blue("free") << "(" << rng.size()
                      << " [" << rng.size()*sizeof(value_type) << " bytes])"
                      << " @ " << rng.data()
                      << std::endl;

        #endif

            allocator.deallocate(rng.data(), rng.size());
        }

        impl::reset(rng);
    }

    // copy memory between host memory ranges
    template <typename Allocator1, typename Allocator2>
    // requires Allocator1 is Allocator
    // requires Allocator2 is Allocator
    void copy(
        const_array_view<value_type, host_coordinator<value_type, Allocator1>> from,
        array_view<value_type, host_coordinator<value_type, Allocator2>> to)
    {
        arb_assert(from.size()==to.size());
        arb_assert(!from.overlaps(to));

        #ifdef VERBOSE
        using c1 = host_coordinator<value_type, Allocator1>;
        std::cerr << util::type_printer<c1>::print()
                  << "::" + util::blue("copy") << "(" << from.size()
                  << " [" << from.size()*sizeof(value_type) << " bytes])"
                  << " " << util::print_pointer(from.data()) << util::yellow(" -> ") << util::print_pointer(to.data())
                  << std::endl;
        #endif

        std::copy(from.begin(), from.end(), to.begin());
    }

    // copy memory from device to host
    template <class Alloc>
    void copy(
        const_array_view<value_type, device_coordinator<value_type, Alloc>> from,
        view_type to)
    {
        arb_assert(from.size()==to.size());

        #ifdef VERBOSE
        std::cerr << util::type_printer<host_coordinator>::print()
                  << "::" + util::blue("copy") << "(device2host, " << from.size()
                  << " [" << from.size()*sizeof(value_type) << " bytes])"
                  << " " << util::print_pointer(from.data()) << util::yellow(" -> ")
                  << util::print_pointer(to.data()) << std::endl;
        #endif

        gpu_memcpy_d2h(to.data(), from.data(), from.size()*sizeof(value_type));
    }

    // copy memory from host to device
    template <class Alloc>
    void copy(
        const_view_type from,
        array_view<value_type, device_coordinator<value_type, Alloc>> to)
    {
        arb_assert(from.size()==to.size());

        #ifdef VERBOSE
        std::cerr << util::type_printer<host_coordinator>::print()
                  << "::" + util::blue("copy") << "(host2device, " << from.size()
                  << " [" << from.size()*sizeof(value_type) << " bytes])"
                  << " " << util::print_pointer(from.data()) << util::yellow(" -> ")
                  << util::print_pointer(to.data()) << std::endl;
        #endif

        gpu_memcpy_h2d(to.data(), from.data(), from.size()*sizeof(value_type));
    }

    // set all values in a range to val
    void set(view_type rng, value_type val) {
        #ifdef VERBOSE
        std::cerr << util::type_printer<host_coordinator>::print()
                  << "::" + util::blue("fill")
                  << "(" << rng.size()  << " * " << val << ")"
                  << " @ " << rng.data()
                  << std::endl;
        #endif
        std::fill(rng.begin(), rng.end(), val);
    }

    reference make_reference(value_type* p) {
        return *p;
    }

    const_reference make_reference(value_type const* p) const {
        return *p;
    }

    static constexpr auto
    alignment() {
        return Allocator::alignment();
    }

    static constexpr bool
    is_malloc_compatible() {
        return Allocator::is_malloc_compatible();
    }
};

} // namespace memory
} // namespace arb
