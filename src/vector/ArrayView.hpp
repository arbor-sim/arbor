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

template <typename R, typename T, typename Coord>
class ArrayViewImpl;

template <typename T, typename Coord>
class ArrayReference;

template <typename R, typename T, typename Coord>
class ConstArrayViewImpl;

template <typename T, typename Coord>
class ConstArrayReference;

namespace util {
    template <typename T, typename Coord>
    struct type_printer<ArrayReference<T,Coord>> {
        static std::string print() {
            std::stringstream str;
            str << util::white("ArrayReference") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct type_printer<ConstArrayReference<T,Coord>> {
        static std::string print() {
            std::stringstream str;
            str << util::white("ConstArrayReference") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<ArrayReference<T,Coord>> {
        static std::string print(const ArrayReference<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<ArrayReference<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << util::print_pointer(val.data()) << ")";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<ConstArrayReference<T,Coord>> {
        static std::string print(const ConstArrayReference<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<ConstArrayReference<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << util::print_pointer(val.data()) << ")";
            return str.str();
        }
    };

    template <typename R, typename T, typename Coord>
    struct type_printer<ArrayViewImpl<R, T,Coord>> {
        static std::string print() {
            std::stringstream str;
            str << util::white("ArrayView") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename R, typename T, typename Coord>
    struct type_printer<ConstArrayViewImpl<R, T,Coord>> {
        static std::string print() {
            std::stringstream str;
            str << util::white("ConstArrayView") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename R, typename T, typename Coord>
    struct pretty_printer<ArrayViewImpl<R, T,Coord>> {
        static std::string print(const ArrayViewImpl<R, T,Coord>& val) {
            std::stringstream str;
            str << type_printer<ArrayViewImpl<R, T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << util::print_pointer(val.data()) << ")";
            return str.str();
        }
    };

    template <typename R, typename T, typename Coord>
    struct pretty_printer<ConstArrayViewImpl<R, T,Coord>> {
        static std::string print(const ConstArrayViewImpl<R, T,Coord>& val) {
            std::stringstream str;
            str << type_printer<ConstArrayViewImpl<R, T,Coord>>::print()
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

    // two specializations for is_array_view
    //      1. for ArrayViewImpl (i.e. ArrayView)
    //      2. for ArrayReference
    template <typename R, typename T, typename Coord>
    struct is_array_view<ArrayViewImpl<R, T, Coord> > : std::true_type {};

    template <typename T, typename Coord>
    struct is_array_view<ArrayReference<T, Coord> > : std::true_type {};

    // and two for const views
    template <typename R, typename T, typename Coord>
    struct is_array_view<ConstArrayViewImpl<R, T, Coord> > : std::true_type {};

    template <typename T, typename Coord>
    struct is_array_view<ConstArrayReference<T, Coord> > : std::true_type {};

    // metafunction for indicating whether a type is an array reference
    template <typename T>
    struct is_array_reference : std::false_type {};

    template <typename T, typename Coord>
    struct is_array_reference<ArrayReference<T, Coord> > : std::true_type {};

    template <typename T, typename Coord>
    struct is_array_reference<ConstArrayReference<T, Coord> > : std::true_type {};

    // metafunction for indicating whether a type is non owning
    // i.e. either a view or a reference
    template <typename T>
    struct is_array_by_reference :
        std::conditional<
            impl::is_array_reference <typename std::decay<T>::type>::value ||
            impl::is_array_view      <typename std::decay<T>::type>::value,
            std::true_type, std::false_type
        >::type
    {};

    template <typename T>
    struct is_array;

    // Helper functions that access the reset() methods in ArrayView.
    // Required to work around a bug in nvcc that makes it awkward to give
    // Coordinator classes friend access to ArrayView types, so that the
    // Coordinator can free and allocate memory. The reset() functions
    // below are friend functions of ArrayView, and are placed in memory::impl::
    // because users should not directly modify pointer or size information in an
    // ArrayView.
    template <typename R, typename T, typename Coord>
    void reset(ArrayViewImpl<R, T, Coord>& v, T* ptr, std::size_t s) {
        v.reset(ptr, s);
    }

    template <typename R, typename T, typename Coord>
    void reset(ArrayViewImpl<R, T, Coord>& v) {
        v.reset();
    }

    template <typename R, typename T, typename Coord>
    void reset(ConstArrayViewImpl<R, T, Coord>& v, T* ptr, std::size_t s) {
        v.reset(ptr, s);
    }

    template <typename R, typename T, typename Coord>
    void reset(ConstArrayViewImpl<R, T, Coord>& v) {
        v.reset();
    }
}

using impl::is_array_by_reference;

// An ArrayView type refers to a sub-range of an Array. It does not own the
// memory, i.e. it is not responsible for allocation and freeing.
// Currently the ArrayRange type has no way of testing whether the memory to
// which it refers is still valid (i.e. whether or not the original memory has
// been freed)
template <typename R, typename T, typename Coord>
class ArrayViewImpl {
public:
    using array_reference_type       = ArrayReference<T, Coord>;
    using const_array_reference_type = ConstArrayReference<T, Coord>;

    using const_view_type = ConstArrayViewImpl<ConstArrayReference<T, Coord>, T, Coord>;

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
    explicit ArrayViewImpl(Other&& other) :
        pointer_(other.data()), size_(other.size())
    {
        #if VERBOSE>1
        std::cout << util::green("ArrayView(&&Other) ")
                  << util::pretty_printer<typename std::decay<Other>::type>::print(*this)
                  << "\n";
        #endif
    }

    explicit ArrayViewImpl(pointer ptr, size_type n) :
        pointer_(ptr), size_(n)
    {
        #if VERBOSE>1
        std::cout << util::green("ArrayView(pointer, size_type)") << util::pretty_printer<ArrayViewImpl>::print(*this) << std::endl;
        #endif
    }

    explicit ArrayViewImpl(ArrayViewImpl& other, size_type n) :
        pointer_(other.data()), size_(n)
    {
        assert(n<=other.size());
        #if VERBOSE>1
        std::cout << util::green("ArrayView(ArrayViewImpl, size_type)") << util::pretty_printer<ArrayViewImpl>::print(*this) << std::endl;
        #endif
    }

    // only works with non const vector until we have a const_view type available
    template <
        typename Allocator,
        typename = typename std::enable_if<coordinator_type::is_malloc_compatible()>::type
     >
    explicit ArrayViewImpl(std::vector<value_type, Allocator>& vec) :
        pointer_(vec.data()), size_(vec.size())
    {
        #if VERBOSE>1
        std::cout << util::green("ArrayView(std::vector)") << util::pretty_printer<ArrayViewImpl>::print(*this) << std::endl;
        #endif
    }

    explicit ArrayViewImpl() {
        reset();
    };

    ////////////////////////////////////////////////////////////////////////////
    // accessors
    // overload operator() to provide range based access
    ////////////////////////////////////////////////////////////////////////////

    /// access half open sub-range using two indexes [left, right)
    array_reference_type operator()(size_type left, size_type right) {
        #ifndef NDEBUG
        assert(right<=size_ && left<=right);
        #endif
        return array_reference_type(pointer_+left, right-left);
    }

    const_array_reference_type operator()(size_type left, size_type right) const {
        #ifndef NDEBUG
        assert(right<=size_ && left<=right);
        #endif
        return array_reference_type(pointer_+left, right-left);
    }

    /// access half open sub-range using one index and one-past-the-end [left, end)
    array_reference_type operator()(size_type left, end_type) {
        #ifndef NDEBUG
        assert(left<=size_);
        #endif
        return array_reference_type(pointer_+left, size_-left);
    }

    const_array_reference_type operator()(size_type left, end_type) const {
        #ifndef NDEBUG
        assert(left<=size_);
        #endif
        return array_reference_type(pointer_+left, size_-left);
    }

    /// access entire range using all
    array_reference_type operator() (all_type) {
        return array_reference_type(pointer_, size_);
    }

    const_array_reference_type operator() (all_type) const {
        return array_reference_type(pointer_, size_);
    }

    // access using a Range
    array_reference_type operator()(Range range) {
        size_type left = range.left();
        size_type right = range.right();
        #ifndef NDEBUG
        assert(right<=size_ && left<=right);
        #endif
        return array_reference_type(pointer_+left, right-left);
    }

    const_array_reference_type operator()(Range range) const {
        size_type left = range.left();
        size_type right = range.right();
        #ifndef NDEBUG
        assert(right<=size_ && left<=right);
        #endif
        return array_reference_type(pointer_+left, right-left);
    }

    template <
        typename Other,
        typename = typename std::enable_if< impl::is_array<Other>::value >::type
    >
    ArrayViewImpl operator=(Other&& other) {
        #if VERBOSE
        std::cerr << util::pretty_printer<ArrayViewImpl>::print(*this)
                  << "::" << util::blue("operator=") << "("
                  << util::pretty_printer<ArrayViewImpl>::print(other)
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
    ~ArrayViewImpl() {}

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
    template <typename RR, typename U, typename C>
    friend void impl::reset(ArrayViewImpl<RR, U, C>&, U*, std::size_t);
    template <typename RR, typename U, typename C>
    friend void impl::reset(ArrayViewImpl<RR, U, C>&);

    void swap(ArrayViewImpl& other) {
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
    ArrayViewImpl(std::size_t n) = delete;

    coordinator_type coordinator_;
    pointer          pointer_;
    size_type        size_;
};

template <typename R, typename T, typename Coord>
class ConstArrayViewImpl {
public:
    using const_array_reference_type = R;

    using view_type = ArrayViewImpl<R, T, Coord>;
    using const_view_type = ConstArrayViewImpl;

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
    ConstArrayViewImpl(Other&& other) :
        pointer_(other.data()), size_(other.size())
    {
#if VERBOSE
        std::cout << util::green("ConstArrayView(&&Other)")
                  << "\n  other "
                  << util::pretty_printer<typename std::decay<Other>::type>::print(other)
                  << std::endl;
#endif
    }

    explicit ConstArrayViewImpl(const_pointer ptr, size_type n) :
        pointer_(ptr), size_(n)
    {
#if VERBOSE
        std::cout << util::green("ConstArrayView(pointer, size_type)")
                  << "\n  other " << util::pretty_printer<ConstArrayViewImpl>::print(*this)
                  << std::endl;
#endif
    }

    explicit ConstArrayViewImpl(view_type& other, size_type n) :
        pointer_(other.data()), size_(n)
    {
        assert(n<=other.size());
    }

    explicit ConstArrayViewImpl(ConstArrayViewImpl& other, size_type n) :
        pointer_(other.data()), size_(n)
    {
        assert(n<=other.size());
    }

    // only works with non const vector until we have a const_view type available
    template <
        typename Allocator,
        typename = typename std::enable_if<coordinator_type::is_malloc_compatible()>::type >
    explicit ConstArrayViewImpl(const std::vector<value_type, Allocator>& vec) :
        pointer_(vec.data()), size_(vec.size())
    {}


    explicit ConstArrayViewImpl() {
        reset();
    };

    ////////////////////////////////////////////////////////////////////////////
    // ACCESSORS
    // overload operator() to provide range based access
    ////////////////////////////////////////////////////////////////////////////

    /// access half open sub-range using two indexes [left, right)
    const_array_reference_type operator()(size_type left, size_type right) const {
#ifndef NDEBUG
        assert(right<=size_ && left<=right);
#endif
        return const_array_reference_type(pointer_+left, right-left);
    }

    /// access half open sub-range using one index and one-past-the-end [left, end)
    const_array_reference_type operator()(size_type left, end_type) const {
#ifndef NDEBUG
        assert(left<=size_);
#endif
        return const_array_reference_type(pointer_+left, size_-left);
    }

    /// access entire range using all
    const_array_reference_type operator() (all_type) const {
        return const_array_reference_type(pointer_, size_);
    }

    // access using a Range
    const_array_reference_type operator()(Range range) const {
        size_type left = range.left();
        size_type right = range.right();
#ifndef NDEBUG
        assert(right<=size_ && left<=right);
#endif
        return const_array_reference_type(pointer_+left, right-left);
    }

    template <
        typename Other,
        typename = typename std::enable_if< impl::is_array<Other>::value >::type
    >
    ConstArrayViewImpl operator=(Other&& other) {
#if VERBOSE
        std::cerr << util::pretty_printer<ConstArrayViewImpl>::print(*this)
                  << "::" << util::blue("operator=") << "("
                  << util::pretty_printer<ConstArrayViewImpl>::print(other)
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
    ~ConstArrayViewImpl() {}

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
    ConstArrayViewImpl(const std::size_t &n) = delete;

protected :
    template <typename RR, typename U, typename C>
    friend void impl::reset(ConstArrayViewImpl<RR, U, C>&, U*, std::size_t);
    template <typename RR, typename U, typename C>
    friend void impl::reset(ConstArrayViewImpl<RR, U, C>&);

    void swap(ConstArrayViewImpl& other) {
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T, typename Coord>
class ArrayReference : public ArrayViewImpl<ArrayReference<T, Coord>, T, Coord> {
public:
    using base = ArrayViewImpl<ArrayReference<T, Coord>, T, Coord>;
    using const_view_type = typename base::const_view_type;
    using value_type = typename base::value_type;
    using size_type  = typename base::size_type;

    using pointer         = typename base::pointer;
    using reference       = typename base::reference;

    using coordinator_type = typename base::coordinator_type;

    using base::coordinator_;
    using base::pointer_;
    using base::size_;
    using base::size;

    // Make only one valid constructor, for maintenance reasons.
    // The only place where ArrayReference types are created is in the
    // operator() calls in the ArrayViewImpl, so it is not an issue to have
    // one method for creating References
    explicit ArrayReference(pointer ptr, size_type n)
        : base(ptr, n)
    {
#if VERBOSE>1
        std::cout << util::green("ArrayReference(pointer, size_type)")
                  << util::pretty_printer<base>::print(*this)
                  << std::endl;
#endif
    }

    // the operator=() operators are key: they facilitate copying data from
    // you can make these return an event type, for synchronization
    template <
        typename Other,
        typename = typename std::enable_if< impl::is_array<Other>::value >::type
    >
    ArrayReference operator = (Other&& other) {
#ifndef NDEBUG
        assert(other.size() == this->size());
#endif
#ifdef VERBOSE
        std::cerr << util::type_printer<ArrayReference>::print()
                  << "::" << util::blue("operator=") << "(&&"
                  << util::type_printer<typename std::decay<Other>::type>::print()
                  << ")" << std::endl;
#endif
        using cvt = typename std::decay<Other>::type::const_view_type;
        base::coordinator_.copy(cvt(other), *this);

        return *this;
    }

    ArrayReference& operator = (value_type value) {
#ifdef VERBOSE
        std::cerr << util::pretty_printer<ArrayReference>::print(*this)
                  << "::" << util::blue("operator=") << "(" << value << ")"
                  << std::endl;
#endif
        if(size()>0) {
            base::coordinator_.set(*this, value);
        }

        return *this;
    }

    template <
        typename Allocator,
        typename = typename
            std::enable_if<coordinator_type::is_malloc_compatible()>::type
     >
    ArrayReference& operator=(const std::vector<value_type, Allocator>& other)
    {
        assert(other.size()==size());
#if VERBOSE>1
        std::cerr << util::pretty_printer<ArrayReference>::print(*this)
                  << "::" << util::blue("operator=") << "(std::vector(" << other.size() << "))"
                  << std::endl;
#endif
        // there is a kink in the coordinator, so use std::copy directly, because we know
        // that coordinator_type is_malloc_compatible
        std::copy(other.begin(), other.end(), base::data());
        return *this;
    }

    // A reference can't be default initialized, because they are designed
    // to be temporary objects that facilitate writing to or reading from
    // memory. Given this, a reference may only be initialized to refer to
    // an existing ArrayView, and it makes no sense to allow one to be default
    // initialized with null data
    ArrayReference() = delete;
};

template <typename T, typename Coord>
class ConstArrayReference : public ConstArrayViewImpl<ConstArrayReference<T, Coord>, T, Coord> {
public:
    using base = ConstArrayViewImpl<ConstArrayReference<T, Coord>, T, Coord>;

    using const_view_type = typename base::const_view_type;

    using value_type = typename base::value_type;
    using size_type  = typename base::size_type;

    using array_reference_type = ArrayReference<T, Coord>;

    using const_pointer   = typename base::const_pointer;
    using const_reference = typename base::const_reference;

    using base::coordinator_;
    using base::pointer_;
    using base::size_;
    using base::size;

    // Make only one valid constructor, for maintenance reasons.
    // The only place where ArrayReference types are created is in the
    // operator() calls in the ArrayViewImpl, so it is not an issue to have
    // one method for creating References
    explicit ConstArrayReference(const_pointer ptr, size_type n)
    : base(ptr, n)
    {
#if VERBOSE
        std::cout << util::green("ConstArrayReference(pointer, size_type)")
                  << util::pretty_printer<base>::print(*this)
                  << std::endl;
#endif
    }

    /// construct a const view from a non-const view
    ConstArrayReference(array_reference_type other)
    : ConstArrayReference(other.pointer_, other.size_)
    { }

    // A reference can't be default initialized, because they are designed
    // to be temporary objects that facilitate writing to or reading from
    // memory. Given this, a reference may only be initialized to refer to
    // an existing ArrayView, and it makes no sense to allow one to be default
    // initialized with null data
    ConstArrayReference() = delete;
};

// export is_array_view helper
using impl::is_array_view;

// export wrappers that define Views in a without the Impl noise
template <typename T, typename Coord>
using ArrayView = ArrayViewImpl<ArrayReference<T, Coord>, T, Coord>;

template <typename T, typename Coord>
using ConstArrayView = ConstArrayViewImpl<ConstArrayReference<T, Coord>, T, Coord>;

} // namespace memory
////////////////////////////////////////////////////////////////////////////////

