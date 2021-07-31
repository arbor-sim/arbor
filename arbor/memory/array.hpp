/*
 * Author: bcumming
 *
 * Created on May 3, 2014, 5:14 PM
 */

#pragma once

#include <cstdlib>
#include <iostream>
#include <type_traits>

#include <arbor/assert.hpp>

#include <util/range.hpp>

#include "definitions.hpp"
#include "util.hpp"
#include "allocator.hpp"
#include "array_view.hpp"

namespace arb {
namespace memory {

// forward declarations
template <typename T, typename Coord>
class array;

template <typename T, class Allocator>
class host_coordinator;

namespace util {
    template <typename T, typename Coord>
    struct type_printer<array<T,Coord>>{
        static std::string print() {
            std::stringstream str;
#if VERBOSE > 1
            str << util::white("array") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
#else
            str << util::white("array") << "<"
                << type_printer<Coord>::print() << ">";
#endif
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<array<T,Coord>>{
        static std::string print(const array<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<array<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << util::print_pointer(val.data()) << ")";
            return str.str();
        }
    };
}

namespace impl {
    // metafunctions for checking array types
    template <typename T>
    struct is_array_by_value : std::false_type {};

    template <typename T, typename Coord>
    struct is_array_by_value<array<T, Coord> > : std::true_type {};

    template <typename T>
    struct is_array :
        std::conditional<
            impl::is_array_by_value<std::decay_t<T>>::value ||
            impl::is_array_view    <std::decay_t<T>>::value,
            std::true_type, std::false_type
        >::type
    {};

    template <typename T>
    using is_array_t = typename is_array<T>::type;
}

using impl::is_array;

// array by value
// this wrapper owns the memory in the array
// and is responsible for allocating and freeing memory
template <typename T, typename Coord>
class array :
    public array_view<T, Coord> {
public:
    using base       = array_view<T, Coord>;
    using view_type  = base;
    using const_view_type = const_array_view<T, Coord>;

    using typename base::value_type;
    using typename base::coordinator_type;

    using typename base::size_type;
    using typename base::difference_type;

    using typename base::pointer;
    using typename base::const_pointer;

    using typename base::iterator;
    using typename base::const_iterator;

    // TODO what about specialized references for things like GPU memory?
    using reference       = value_type&;
    using const_reference = value_type const&;

    // default constructor
    // create empty storage
    array() :
        base(nullptr, 0)
    {}

    // constructor by size
    template <
        typename I,
        typename = std::enable_if_t<std::is_integral<I>::value>>
    array(I n) :
        base(coordinator_type().allocate(n))
    {
#ifdef VERBOSE
        std::cerr << util::green("array(" + std::to_string(n) + ")")
                  << "\n  this  " << util::pretty_printer<array>::print(*this) << std::endl;
#endif
    }

    // constructor by size with default value
    template <
        typename II, typename TT,
        typename = std::enable_if_t<std::is_integral<II>::value>,
        typename = std::enable_if_t<std::is_convertible<TT,value_type>::value> >
    array(II n, TT value) :
        base(coordinator_type().allocate(n))
    {
        #ifdef VERBOSE
        std::cerr << util::green("array(" + std::to_string(n) + ", " + std::to_string(value) + ")")
                  << "\n  this  " << util::pretty_printer<array>::print(*this) << "\n";
        #endif
        coordinator_type().set(*this, value_type(value));
    }

    // copy constructor
    array(const array& other) :
        base(coordinator_type().allocate(other.size()))
    {
        static_assert(impl::is_array_t<array>::value, "");
#ifdef VERBOSE
        std::cerr << util::green("array(array&)")
                  << " " << util::type_printer<array>::print()
                  << "\n  this  " << util::pretty_printer<array>::print(*this)
                  << "\n  other " << util::pretty_printer<array>::print(other) << "\n";
#endif
        base::coordinator_.copy(const_view_type(other), view_type(*this));
    }

    // move constructor
    array(array&& other) {
#ifdef VERBOSE
        std::cerr << util::green("array(array&&)")
                  << " " << util::type_printer<array>::print()
                  << "\n  other " << util::pretty_printer<array>::print(other) << "\n";
#endif
        base::swap(other);
    }

    // copy constructor where other is an array, array_view or array_reference
    template <
        typename Other,
        typename = std::enable_if_t<impl::is_array_t<Other>::value>
    >
    array(const Other& other) :
        base(coordinator_type().allocate(other.size()))
    {
#ifdef VERBOSE
        std::cerr << util::green("array(Other&)")
                  << " " << util::type_printer<array>::print()
                  << "\n  this  " << util::pretty_printer<array>::print(*this)
                  << "\n  other " << util::pretty_printer<Other>::print(other) << std::endl;
#endif
        base::coordinator_.copy(typename Other::const_view_type(other), view_type(*this));
    }

    array& operator=(const array& other) {
#ifdef VERBOSE
        std::cerr << util::green("array operator=(array&)")
                  << "\n  this  "  << util::pretty_printer<array>::print(*this)
                  << "\n  other " << util::pretty_printer<array>::print(other) << "\n";
#endif
        base::coordinator_.free(*this);
        auto ptr = coordinator_type().allocate(other.size());
        base::reset(ptr.data(), other.size());
        base::coordinator_.copy(const_view_type(other), view_type(*this));
        return *this;
    }

    array& operator = (array&& other) {
#ifdef VERBOSE
        std::cerr << util::green("array operator=(array&&)")
                  << "\n  this  "  << util::pretty_printer<array>::print(*this)
                  << "\n  other " << util::pretty_printer<array>::print(other) << "\n";
#endif
        base::swap(other);
        return *this;
    }

    // have to free the memory in a "by value" range
    ~array() {
#ifdef VERBOSE
        std::cerr << util::red("~") + util::green("array()")
                  << "\n  this " << util::pretty_printer<array>::print(*this) << "\n";
#endif
        base::coordinator_.free(*this);
    }

    template <
        typename It,
        typename = std::enable_if_t<arb::util::is_random_access_iterator<It>::value> >
    array(It b, It e) :
        base(coordinator_type().allocate(std::distance(b, e)))
    {
#ifdef VERBOSE
        std::cerr << util::green("array(iterator, iterator)")
                  << " " << util::type_printer<array>::print()
                  << "\n  this  " << util::pretty_printer<array>::print(*this) << "\n";
                  //<< "\n  other " << util::pretty_printer<Other>::print(other) << std::endl;
#endif
        // Only valid for contiguous range, but we can't test that at compile time.
        // Can check though that taking &*b+n = &*e where n = e-b, while acknowledging
        // this is not fail safe.
        arb_assert(&*b+(e-b)==&*e);

        using V = typename std::iterator_traits<iterator>::value_type;
        base::coordinator_.copy(const_array_view<V, host_coordinator<V, aligned_allocator<V>>>(&*b, e-b), view_type(*this));
    }

    // use the accessors provided by array_view
    // this enforces the requirement that accessing all of or a sub-array of an
    // array should return a view, not a new array.
    using base::operator();

    const coordinator_type& coordinator() const {
        return base::coordinator_;
    }

    using base::size;

    using base::alignment;
};

} // namespace memory
} // namespace arb

