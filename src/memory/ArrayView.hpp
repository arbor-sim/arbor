#pragma once

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "definitions.hpp"
#include "util.hpp"
#include "Range.hpp"
#include "RangeLimits.hpp"

////////////////////////////////////////////////////////////////////////////////
namespace memory{

// forward declarations
template<typename T, typename Coord>
class Array;

template <typename T, typename Coord>
class ArrayView;

template <typename T, typename Coord>
class ConstArrayView;

namespace util {
    template <typename T, typename Coord>
    struct type_printer<ArrayView<T,Coord>> {
        static std::string print() {
            std::stringstream str;
            str << util::white("ArrayView") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct type_printer<ConstArrayView<T,Coord>> {
        static std::string print() {
            std::stringstream str;
            str << util::white("ConstArrayView") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<ArrayView<T,Coord>> {
        static std::string print(const ArrayView<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<ArrayView<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << util::print_pointer(val.data()) << ")";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<ConstArrayView<T,Coord>> {
        static std::string print(const ConstArrayView<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<ConstArrayView<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << util::print_pointer(val.data()) << ")";
            return str.str();
        }
    };
}

namespace impl {

    // metafunction for indicating whether a type is an ArrayView
    template <typename T>
    struct is_array_view : std::false_type {};

    template <typename T, typename Coord>
    struct is_array_view<ArrayView<T, Coord> > : std::true_type {};

    template <typename T, typename Coord>
    struct is_array_view<ConstArrayView<T, Coord> > : std::true_type {};

    template <typename T>
    struct is_array;

    // Helper functions that access the reset() methods in ArrayView.
    // Required to work around a bug in nvcc that makes it awkward to give
    // Coordinator classes friend access to ArrayView types, so that the
    // Coordinator can free and allocate memory. The reset() functions
    // below are friend functions of ArrayView, and are placed in memory::impl::
    // because users should not directly modify pointer or size information in an
    // ArrayView.
    template <typename T, typename Coord>
    void reset(ArrayView<T, Coord>& v, T* ptr, std::size_t s) {
        v.reset(ptr, s);
    }

    template <typename T, typename Coord>
    void reset(ArrayView<T, Coord>& v) {
        v.reset();
    }

    template <typename T, typename Coord>
    void reset(ConstArrayView<T, Coord>& v, T* ptr, std::size_t s) {
        v.reset(ptr, s);
    }

    template <typename T, typename Coord>
    void reset(ConstArrayView<T, Coord>& v) {
        v.reset();
    }
}

using impl::is_array_view;

// An ArrayView type refers to a sub-range of an Array. It does not own the
// memory, i.e. it is not responsible for allocation and freeing.
template <typename T, typename Coord>
class ArrayView {
public:
    using view_type       = ArrayView<T, Coord>;
    using const_view_type = ConstArrayView<T, Coord>;

    using value_type = T;
    using coordinator_type = typename Coord::template rebind<value_type>;

    using size_type       = types::size_type;
    using difference_type = types::difference_type;

    using pointer         = typename coordinator_type::pointer;
    using const_pointer   = typename coordinator_type::const_pointer;
    using reference       = typename coordinator_type::reference;
    using const_reference = typename coordinator_type::const_reference;

    ////////////////////////////////////////////////////////////////////////////
    // constructors
    ////////////////////////////////////////////////////////////////////////////
    template <
        typename Other,
        typename = typename std::enable_if< impl::is_array<Other>::value >::type
    >
    explicit ArrayView(Other&& other) :
        pointer_(other.data()), size_(other.size())
    {
        #ifdef VERBOSE
        std::cout << util::green("ArrayView(&&Other) ")
                  << "\n  this  " << util::pretty_printer<ArrayView>::print(*this)
                  << "\n  other " << util::pretty_printer<typename std::decay<Other>::type>::print(other)
                  << "\n";
        #endif
    }

    explicit ArrayView(pointer ptr, size_type n) :
        pointer_(ptr), size_(n)
    {
        #ifdef VERBOSE
        std::cout << util::green("ArrayView(pointer, size_type)")
                  << "\n  this  " << util::pretty_printer<ArrayView>::print(*this)
                  << "\n";
        #endif
    }

    explicit ArrayView(ArrayView& other, size_type n) :
        pointer_(other.data()), size_(n)
    {
        assert(n<=other.size());
        #ifdef VERBOSE
        std::cout << util::green("ArrayView(ArrayView, size_type)")
                  << "\n  this  " << util::pretty_printer<ArrayView>::print(*this)
                  << "\n  other " << util::pretty_printer<ArrayView>::print(other) << "\n";
        #endif
    }

    explicit ArrayView() {
        reset();
    };

    ////////////////////////////////////////////////////////////////////////////
    // accessors
    // overload operator() to provide range based access
    ////////////////////////////////////////////////////////////////////////////

    /// access half open sub-range using two indexes [left, right)
    view_type operator()(size_type left, size_type right) {
        #ifndef NDEBUG
        assert(right<=size_ && left<=right);
        #endif
        return view_type(pointer_+left, right-left);
    }

    const_view_type operator()(size_type left, size_type right) const {
        #ifndef NDEBUG
        assert(right<=size_ && left<=right);
        #endif
        return view_type(pointer_+left, right-left);
    }

    /// access half open sub-range using one index and one-past-the-end [left, end)
    view_type operator()(size_type left, end_type) {
        #ifndef NDEBUG
        assert(left<=size_);
        #endif
        return view_type(pointer_+left, size_-left);
    }

    const_view_type operator()(size_type left, end_type) const {
        #ifndef NDEBUG
        assert(left<=size_);
        #endif
        return view_type(pointer_+left, size_-left);
    }

    /// access entire range using all
    view_type operator() (all_type) {
        return view_type(pointer_, size_);
    }

    const_view_type operator() (all_type) const {
        return view_type(pointer_, size_);
    }

    // access using a Range
    view_type operator()(Range range) {
        size_type left = range.left();
        size_type right = range.right();
        #ifndef NDEBUG
        assert(right<=size_ && left<=right);
        #endif
        return view_type(pointer_+left, right-left);
    }

    const_view_type operator()(Range range) const {
        size_type left = range.left();
        size_type right = range.right();
        #ifndef NDEBUG
        assert(right<=size_ && left<=right);
        #endif
        return view_type(pointer_+left, right-left);
    }

    template <
        typename Other,
        typename = typename std::enable_if< impl::is_array<Other>::value >::type
    >
    ArrayView operator=(Other&& other) {
        #if VERBOSE
        std::cerr << util::pretty_printer<ArrayView>::print(*this)
                  << "::" << util::blue("operator=") << "("
                  << util::pretty_printer<ArrayView>::print(other)
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
        #ifndef NDEBUG
        assert(i<size_);
        #endif
        return coordinator_.make_reference(pointer_+i);
    }

    const_reference operator[] (size_type i) const {
        #ifndef NDEBUG
        assert(i<size_);
        #endif
        return coordinator_.make_reference(pointer_+i);
    }

    // do nothing for destructor: we don't own the memory in range
    ~ArrayView() {}

    // test whether memory overlaps that referenced by other
    template <
        typename Other,
        typename = typename std::enable_if<impl::is_array<Other>::value>
    >
    bool overlaps(Other&& other) const {
        return( !((this->begin()>=other.end()) || (other.begin()>=this->end())) );
    }

    memory::Range range() const {
        return memory::Range(0, size());
    }

    static constexpr auto
    alignment() -> decltype(coordinator_type::alignment()) {
        return coordinator_type::alignment();
    }

protected :
    template < typename U, typename C>
    friend void impl::reset(ArrayView<U, C>&, U*, std::size_t);
    template <typename U, typename C>
    friend void impl::reset(ArrayView<U, C>&);

    void swap(ArrayView& other) {
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
    ArrayView(std::size_t n) = delete;

    coordinator_type coordinator_;
    pointer          pointer_;
    size_type        size_;
};

template <typename T, typename Coord>
class ConstArrayView {
public:
    using view_type       = ArrayView<T, Coord>;
    using const_view_type = ConstArrayView;

    using value_type = T;
    using coordinator_type = typename Coord::template rebind<value_type>;

    using size_type       = types::size_type;
    using difference_type = types::difference_type;

    using const_pointer   = typename coordinator_type::const_pointer;
    using const_reference = typename coordinator_type::const_reference;

    ////////////////////////////////////////////////////////////////////////////
    // CONSTRUCTORS
    ////////////////////////////////////////////////////////////////////////////
    template <
        typename Other,
        typename = typename std::enable_if< impl::is_array<Other>::value >::type
    >
    ConstArrayView(Other&& other) :
        pointer_(other.data()), size_(other.size())
    {
#if VERBOSE
        std::cout << util::green("ConstArrayView(&&Other)")
                  << "\n  this  " << util::pretty_printer<ConstArrayView>::print(*this)
                  << "\n  other " << util::pretty_printer<typename std::decay<Other>::type>::print(other)
                  << std::endl;
#endif
    }

    explicit ConstArrayView(const_pointer ptr, size_type n) :
        pointer_(ptr), size_(n)
    {
#if VERBOSE
        std::cout << util::green("ConstArrayView(pointer, size_type)")
                  << "\n  this " << util::pretty_printer<ConstArrayView>::print(*this)
                  << std::endl;
#endif
    }

    explicit ConstArrayView(view_type& other, size_type n) :
        pointer_(other.data()), size_(n)
    {
        assert(n<=other.size());
    }

    explicit ConstArrayView(ConstArrayView& other, size_type n) :
        pointer_(other.data()), size_(n)
    {
        assert(n<=other.size());
    }

    explicit ConstArrayView() {
        reset();
    };

    ////////////////////////////////////////////////////////////////////////////
    // ACCESSORS
    // overload operator() to provide range based access
    ////////////////////////////////////////////////////////////////////////////

    /// access half open sub-range using two indexes [left, right)
    const_view_type operator()(size_type left, size_type right) const {
#ifndef NDEBUG
        assert(right<=size_ && left<=right);
#endif
        return const_view_type(pointer_+left, right-left);
    }

    /// access half open sub-range using one index and one-past-the-end [left, end)
    const_view_type operator()(size_type left, end_type) const {
#ifndef NDEBUG
        assert(left<=size_);
#endif
        return const_view_type(pointer_+left, size_-left);
    }

    /// access entire range using all
    const_view_type operator() (all_type) const {
        return const_view_type(pointer_, size_);
    }

    // access using a Range
    const_view_type operator()(Range range) const {
        size_type left = range.left();
        size_type right = range.right();
#ifndef NDEBUG
        assert(right<=size_ && left<=right);
#endif
        return const_view_type(pointer_+left, right-left);
    }

    template <
        typename Other,
        typename = typename std::enable_if< impl::is_array<Other>::value >::type
    >
    ConstArrayView operator=(Other&& other) {
#if VERBOSE
        std::cerr << util::pretty_printer<ConstArrayView>::print(*this)
                  << "::" << util::blue("operator=") << "("
                  << util::pretty_printer<ConstArrayView>::print(other)
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
        #ifndef NDEBUG
        assert(i<size_);
        #endif
        return coordinator_.make_reference(pointer_+i);
    }

    // do nothing for destructor: we don't own the memory in range
    ~ConstArrayView() {}

    // test whether memory overlaps that referenced by other
    template <
        typename Other,
        typename = typename std::enable_if<impl::is_array<Other>::value>
    >
    bool overlaps(Other&& other) const {
        return( !((this->begin()>=other.end()) || (other.begin()>=this->end())) );
    }

    memory::Range range() const {
        return memory::Range(0, size());
    }

    static constexpr auto
    alignment() -> decltype(coordinator_type::alignment()) {
        return coordinator_type::alignment();
    }

    // disallow constructors that imply allocation of memory
    ConstArrayView(const std::size_t &n) = delete;

protected :
    template <typename U, typename C>
    friend void impl::reset(ConstArrayView<U, C>&, U*, std::size_t);
    template <typename U, typename C>
    friend void impl::reset(ConstArrayView<U, C>&);

    void swap(ConstArrayView& other) {
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
////////////////////////////////////////////////////////////////////////////////

