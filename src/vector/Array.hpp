/*
 * File:   range_wrapper.h
 * Author: bcumming
 *
 * Created on May 3, 2014, 5:14 PM
 */

#pragma once

#include <iostream>
#include <type_traits>

#include "definitions.hpp"
#include "util.hpp"
#include "ArrayView.hpp"

////////////////////////////////////////////////////////////////////////////////
namespace memory{

// forward declarations
template<typename T, typename Coord>
class Array;

namespace util {
    template <typename T, typename Coord>
    struct type_printer<Array<T,Coord>>{
        static std::string print() {
            std::stringstream str;
#if VERBOSE > 1
            str << util::white("Array") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
#else
            str << util::white("Array") << "<"
                << type_printer<Coord>::print() << ">";
#endif
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<Array<T,Coord>>{
        static std::string print(const Array<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<Array<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << val.data() << ")";
            return str.str();
        }
    };
}

namespace impl {
    // metafunctions for checking array types
    template <typename T>
    struct is_array_by_value : std::false_type {};

    template <typename T, typename Coord>
    struct is_array_by_value<Array<T, Coord> > : std::true_type {};

    template <typename T>
    struct is_array :
        std::conditional<
            impl::is_array_by_value  <typename std::decay<T>::type> ::value ||
            impl::is_array_view      <typename std::decay<T>::type> ::value ||
            impl::is_array_reference <typename std::decay<T>::type> ::value,
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
class Array
    : public ArrayView<T, Coord> {
public:
    using value_type = T;
    using base       = ArrayView<value_type, Coord>;
    using view_type  = ArrayView<value_type, Coord>;
    using const_view_type  = ConstArrayView<value_type, Coord>;

    using coordinator_type = typename Coord::template rebind<value_type>;

    using size_type       = typename base::size_type;
    using difference_type = typename base::difference_type;

    using pointer       = value_type*;
    using const_pointer = const pointer;
    // TODO what about specialized references for things like GPU memory?
    using reference       = value_type&;
    using const_reference = value_type const&;

    // default constructor
    // we have to call constructor in ArrayView: pass base
    Array() : base(nullptr, 0) {}

    // constructor by size
    template < typename I,
               typename = typename std::enable_if<std::is_integral<I>::value>::type>
    Array(I n)
        : base(coordinator_type().allocate(n))
    {
        #ifdef VERBOSE
        std::cerr << util::green("Array(integral_type) ")
                  << util::pretty_printer<Array>::print(*this) << std::endl;
        #endif
    }

    // constructor by size with default value
    template < typename II,
               typename TT,
               typename = typename std::enable_if<std::is_integral<II>::value>::type,
               typename = typename std::enable_if<std::is_convertible<TT,value_type>::value>::type >
    Array(II n, TT value)
        : base(coordinator_type().allocate(n))
    {
        #ifdef VERBOSE
        std::cerr << util::green("Array(integral_type, value=" + std::to_string(value) + ") ")
                  << util::pretty_printer<Array>::print(*this) << std::endl;
        #endif
        coordinator_type().set(*this, value_type(value));
    }

    template <typename Other,
              typename = typename
                  std::enable_if<
                                 impl::is_array_t<Other>::value
                                >::type
             >
    Array(Other&& other)
        : base(coordinator_type().allocate(other.size()))
    {
        coordinator_.copy(other, *this);
    }

    // construct as a copy of another range
    Array(view_type const& other)
        : base(coordinator_type().allocate(other.size()))
    {
#ifdef VERBOSE
        std::cerr << util::green("Array(other&)") + " other = "
                  << util::pretty_printer<view_type>::print(other) << std::endl;
#endif
        coordinator_.copy(static_cast<base const&>(other), *this);
    }

    Array(const Array& other)
        : base(coordinator_type().allocate(other.size()))
    {
#ifdef VERBOSE
        std::cerr << util::green("Array(other&)") + " other = "
                  << util::pretty_printer<Array>::print(other) << std::endl;
#endif
        coordinator_.copy(static_cast<base const&>(other), *this);
    }

    Array(Array&& other) {
#ifdef VERBOSE
        std::cerr << util::green("Array(Array&&) ")
                  << util::pretty_printer<Array>::print(other) << std::endl;
#endif
        base::swap(other);
    }

    /// copy from a std::vector
    /// the value_type of the vector must be the same, because the coordinator
    /// used to copy from the vector into the Array does not convert between types
    template < typename Allocator >
    Array(std::vector<value_type, Allocator> const& other)
    : base(coordinator_type().allocate(other.size()))
    {
        coordinator_.copy(
            const_view_type(other.data(), other.size()),
            *this
        );
    }

    Array& operator = (Array const& other) {
#ifdef VERBOSE
        std::cerr << util::green("Array operator=(other&)") + " other = "
                  << util::pretty_printer<Array>::print(other) << std::endl;
#endif
        coordinator_.free(*this);
        auto ptr = coordinator_type().allocate(other.size());
        base::reset(ptr.data(), other.size());
        coordinator_.copy(static_cast<base const&>(other), *this);
        return *this;
    }

    Array& operator = (Array&& other) {
#ifdef VERBOSE
        std::cerr << util::pretty_printer<Array>::print(*this)
                  << "::" << util::blue("operator=") << "(Array&&) other = "
                  << util::pretty_printer<Array>::print(other) << std::endl;
#endif
        base::swap(other);
        return *this;
    }

    // have to free the memory in a "by value" range
    ~Array() {
#ifdef VERBOSE
        std::cerr << util::red("~") + util::green("Array()") + " "
                  << util::pretty_printer<Array>::print(*this) << std::endl;
#endif
        coordinator_.free(*this);
    }

    // use the accessors provided by ArrayView
    // this enforces the requirement that accessing all of or a sub-array of an
    // Array should return a view, not a new array.
    using base::operator();

    const coordinator_type& coordinator() const {
        return coordinator_;
    }

    using base::size;

    using base::alignment;

private:
    coordinator_type coordinator_;
};

} // namespace memory
////////////////////////////////////////////////////////////////////////////////

