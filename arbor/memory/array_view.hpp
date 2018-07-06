#pragma once

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include <arbor/assert.hpp>

#include "definitions.hpp"
#include "util.hpp"
#include "range_limits.hpp"

namespace arb {
namespace memory{

// forward declarations
template<typename T, typename Coord>
class array;

template <typename T, typename Coord>
class array_view;

template <typename T, typename Coord>
class const_array_view;

namespace util {
    template <typename T, typename Coord>
    struct type_printer<array_view<T,Coord>> {
        static std::string print() {
            std::stringstream str;
            str << util::white("array_view") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct type_printer<const_array_view<T,Coord>> {
        static std::string print() {
            std::stringstream str;
            str << util::white("const_array_view") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<array_view<T,Coord>> {
        static std::string print(const array_view<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<array_view<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << util::print_pointer(val.data()) << ")";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<const_array_view<T,Coord>> {
        static std::string print(const const_array_view<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<const_array_view<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << util::print_pointer(val.data()) << ")";
            return str.str();
        }
    };
}

namespace impl {

    // metafunction for indicating whether a type is an array_view
    template <typename T>
    struct is_array_view : std::false_type {};

    template <typename T, typename Coord>
    struct is_array_view<array_view<T, Coord> > : std::true_type {};

    template <typename T, typename Coord>
    struct is_array_view<const_array_view<T, Coord> > : std::true_type {};

    template <typename T>
    struct is_array;

    // Helper functions that access the reset() methods in array_view.
    // Required to work around a bug in nvcc that makes it awkward to give
    // Coordinator classes friend access to array_view types, so that the
    // Coordinator can free and allocate memory. The reset() functions
    // below are friend functions of array_view, and are placed in memory::impl::
    // because users should not directly modify pointer or size information in an
    // array_view.
    template <typename T, typename Coord>
    void reset(array_view<T, Coord>& v, T* ptr, std::size_t s) {
        v.reset(ptr, s);
    }

    template <typename T, typename Coord>
    void reset(array_view<T, Coord>& v) {
        v.reset();
    }

    template <typename T, typename Coord>
    void reset(const_array_view<T, Coord>& v, T* ptr, std::size_t s) {
        v.reset(ptr, s);
    }

    template <typename T, typename Coord>
    void reset(const_array_view<T, Coord>& v) {
        v.reset();
    }
}

using impl::is_array_view;

// An array_view type refers to a sub-range of an array. It does not own the
// memory, i.e. it is not responsible for allocation and freeing.
template <typename T, typename Coord>
class array_view {
public:
    using view_type       = array_view<T, Coord>;
    using const_view_type = const_array_view<T, Coord>;

    using value_type = T;
    using coordinator_type = typename Coord::template rebind<value_type>;

    using size_type       = types::size_type;
    using difference_type = types::difference_type;

    using pointer         = typename coordinator_type::pointer;
    using const_pointer   = typename coordinator_type::const_pointer;
    using reference       = typename coordinator_type::reference;
    using const_reference = typename coordinator_type::const_reference;

    using iterator = pointer;
    using const_iterator = const_pointer;

    // Constructors
    template <
        typename Other,
        typename = std::enable_if_t< impl::is_array<Other>::value >
    >
    explicit array_view(Other&& other) :
        pointer_(other.data()), size_(other.size())
    {
        #ifdef VERBOSE
        std::cout << util::green("array_view(&&Other) ")
                  << "\n  this  " << util::pretty_printer<array_view>::print(*this)
                  << "\n  other " << util::pretty_printer<std::decay_t<Other>>::print(other)
                  << "\n";
        #endif
    }

    explicit array_view(pointer ptr, size_type n) :
        pointer_(ptr), size_(n)
    {
        #ifdef VERBOSE
        std::cout << util::green("array_view(pointer, size_type)")
                  << "\n  this  " << util::pretty_printer<array_view>::print(*this)
                  << "\n";
        #endif
    }

    explicit array_view(array_view& other, size_type n) :
        pointer_(other.data()), size_(n)
    {
        arb_assert(n<=other.size());
        #ifdef VERBOSE
        std::cout << util::green("array_view(array_view, size_type)")
                  << "\n  this  " << util::pretty_printer<array_view>::print(*this)
                  << "\n  other " << util::pretty_printer<array_view>::print(other) << "\n";
        #endif
    }

    explicit array_view() {
        reset();
    };

    // Accessors: overload operator() to provide range based access

    /// access half open sub-range using two indexes [left, right)
    view_type operator()(size_type left, size_type right) {
        arb_assert(right<=size_ && left<=right);
        return view_type(pointer_+left, right-left);
    }

    const_view_type operator()(size_type left, size_type right) const {
        arb_assert(right<=size_ && left<=right);
        return view_type(pointer_+left, right-left);
    }

    /// access half open sub-range using one index and one-past-the-end [left, end)
    view_type operator()(size_type left, end_type) {
        arb_assert(left<=size_);
        return view_type(pointer_+left, size_-left);
    }

    const_view_type operator()(size_type left, end_type) const {
        arb_assert(left<=size_);
        return view_type(pointer_+left, size_-left);
    }

    template <
        typename Other,
        typename = std::enable_if_t< impl::is_array<Other>::value >
    >
    array_view operator=(Other&& other) {
        #if VERBOSE
        std::cerr << util::pretty_printer<array_view>::print(*this)
                  << "::" << util::blue("operator=") << "("
                  << util::pretty_printer<array_view>::print(other)
                  << ")" << std::endl;
        #endif
        reset(other.data(), other.size());
        return *this;
    }

    // access to raw data
    pointer data() {
        return pointer_;
    }

    const_pointer data() const {
        return const_pointer(pointer_);
    }

    size_type size() const {
        return size_;
    }

    bool is_empty() const {
        return size_==0;
    }

    // begin/end iterator pairs
    pointer begin() {
        return pointer_;
    }

    const_pointer begin() const {
        return pointer_;
    }

    pointer end() {
        return pointer_+size_;
    }

    const_pointer end() const {
        return pointer_+size_;
    }

    // per element accessors
    // return a reference type provided by Coordinator
    reference operator[] (size_type i) {
        arb_assert(i<size_);
        return coordinator_.make_reference(pointer_+i);
    }

    const_reference operator[] (size_type i) const {
        arb_assert(i<size_);
        return coordinator_.make_reference(pointer_+i);
    }

    // do nothing for destructor: we don't own the memory in range
    ~array_view() {}

    // test whether memory overlaps that referenced by other
    template <
        typename Other,
        typename = typename std::enable_if<impl::is_array<Other>::value>
    >
    bool overlaps(Other&& other) const {
        return( !((this->begin()>=other.end()) || (other.begin()>=this->end())) );
    }

    static constexpr auto
    alignment() {
        return coordinator_type::alignment();
    }

protected :
    template < typename U, typename C>
    friend void impl::reset(array_view<U, C>&, U*, std::size_t);
    template <typename U, typename C>
    friend void impl::reset(array_view<U, C>&);

    void swap(array_view& other) {
        auto ptr = other.data();
        auto sz  = other.size();
        other.reset(pointer_, size_);
        pointer_ = ptr;
        size_    = sz;
    }

    void reset() {
        pointer_ = nullptr;
        size_ = 0;
    }

    void reset(pointer ptr, size_type n) {
        pointer_ = ptr;
        size_ = n;
    }

    // disallow constructors that imply allocation of memory
    array_view(std::size_t n) = delete;

    coordinator_type coordinator_;
    pointer          pointer_;
    size_type        size_;
};

template <typename T, typename Coord>
class const_array_view {
public:
    using view_type       = array_view<T, Coord>;
    using const_view_type = const_array_view;

    using value_type = T;
    using coordinator_type = typename Coord::template rebind<value_type>;

    using size_type       = types::size_type;
    using difference_type = types::difference_type;

    using const_pointer   = typename coordinator_type::const_pointer;
    using const_reference = typename coordinator_type::const_reference;

    using const_iterator = const_pointer;

    // Constructors
    template <
        typename Other,
        typename = std::enable_if_t< impl::is_array<Other>::value >
    >
    const_array_view(const Other& other) :
        pointer_(other.data()), size_(other.size())
    {
#if VERBOSE
        std::cout << util::green("const_array_view(const Other&)")
                  << "\n  this  " << util::pretty_printer<const_array_view>::print(*this)
                  << "\n  other " << util::pretty_printer<std::decay_t<Other>>::print(other)
                  << std::endl;
#endif
    }

    explicit const_array_view(const_pointer ptr, size_type n) :
        pointer_(ptr), size_(n)
    {
#if VERBOSE
        std::cout << util::green("const_array_view(pointer, size_type)")
                  << "\n  this " << util::pretty_printer<const_array_view>::print(*this)
                  << std::endl;
#endif
    }

    explicit const_array_view(view_type& other, size_type n) :
        pointer_(other.data()), size_(n)
    {
        arb_assert(n<=other.size());
    }

    explicit const_array_view(const_array_view& other, size_type n) :
        pointer_(other.data()), size_(n)
    {
        arb_assert(n<=other.size());
    }

    explicit const_array_view() {
        reset();
    };

    // Accessors: overload operator() to provide range based access

    /// access half open sub-range using two indexes [left, right)
    const_view_type operator()(size_type left, size_type right) const {
        arb_assert(right<=size_ && left<=right);
        return const_view_type(pointer_+left, right-left);
    }

    /// access half open sub-range using one index and one-past-the-end [left, end)
    const_view_type operator()(size_type left, end_type) const {
        arb_assert(left<=size_);
        return const_view_type(pointer_+left, size_-left);
    }

    template <
        typename Other,
        typename = std::enable_if_t< impl::is_array<Other>::value >
    >
    const_array_view operator=(Other&& other) {
#if VERBOSE
        std::cerr << util::pretty_printer<const_array_view>::print(*this)
                  << "::" << util::blue("operator=") << "("
                  << util::pretty_printer<const_array_view>::print(other)
                  << ")" << std::endl;
#endif
        reset(other.data(), other.size());
        return *this;
    }

    // access to raw data
    const_pointer data() const {
        return pointer_;
    }

    size_type size() const {
        return size_;
    }

    bool is_empty() const {
        return size_==0;
    }

    // begin/end iterator pairs
    const_pointer begin() const {
        return pointer_;
    }

    const_pointer end() const {
        return pointer_+size_;
    }

    // per element accessors
    // return a reference type provided by Coordinator
    const_reference operator[] (size_type i) const {
        arb_assert(i<size_);
        return coordinator_.make_reference(pointer_+i);
    }

    // do nothing for destructor: we don't own the memory in range
    ~const_array_view() {}

    // test whether memory overlaps that referenced by other
    template <
        typename Other,
        typename = typename std::enable_if<impl::is_array<Other>::value>
    >
    bool overlaps(Other&& other) const {
        return( !((this->begin()>=other.end()) || (other.begin()>=this->end())) );
    }

    static constexpr auto
    alignment() {
        return coordinator_type::alignment();
    }

    // disallow constructors that imply allocation of memory
    const_array_view(const std::size_t &n) = delete;

protected :
    template <typename U, typename C>
    friend void impl::reset(const_array_view<U, C>&, U*, std::size_t);
    template <typename U, typename C>
    friend void impl::reset(const_array_view<U, C>&);

    void swap(const_array_view& other) {
        auto ptr = other.data();
        auto sz  = other.size();
        other.reset(pointer_, size_);
        pointer_ = ptr;
        size_    = sz;
    }

    void reset() {
        pointer_ = nullptr;
        size_ = 0;
    }

    void reset(const_pointer ptr, size_type n) {
        pointer_ = ptr;
        size_ = n;
    }

    coordinator_type coordinator_;
    const_pointer    pointer_;
    size_type        size_;
};

// export is_array_view helper
using impl::is_array_view;

} // namespace memory
} // namespace arb

