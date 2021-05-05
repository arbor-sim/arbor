#pragma once

// Create/manipulate 1-d piece-wise defined objects.
//
// Using vectors everywhere here for ease; consider making
// something more container/sequence-generic later.

#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <vector>

#include "util/meta.hpp"
#include "util/partition.hpp"

namespace arb {
namespace util {

using pw_size_type = unsigned;
constexpr pw_size_type pw_npos = -1;

// Generic random access const iterator for a collection
// providing operator[].

template <typename T>
struct indexed_const_iterator {
    using size_type = decltype(std::size(std::declval<T>()));
    using difference_type = std::make_signed_t<size_type>;

    using value_type = decltype(std::declval<T>()[0]);
    struct pointer {
        value_type v;
        const value_type* operator->() const { return &v; }
    };

    using reference = value_type;
    using iterator_category = std::random_access_iterator_tag;

    const T* ptr_ = nullptr;
    size_type i_ = 0;

    bool operator==(indexed_const_iterator x) const { return ptr_ == x.ptr_ && i_ == x.i_; }
    bool operator!=(indexed_const_iterator x) const { return !(*this==x); }
    bool operator<=(indexed_const_iterator x) const { return i_<=x.i_; }
    bool operator<(indexed_const_iterator x)  const { return i_<x.i_; }
    bool operator>=(indexed_const_iterator x) const { return i_>=x.i_; }
    bool operator>(indexed_const_iterator x)  const { return i_>x.i_; }

    difference_type operator-(indexed_const_iterator x) const { return i_-x.i_; }
    indexed_const_iterator& operator++() { return ++i_, *this; }
    indexed_const_iterator& operator--() { return --i_, *this; }
    indexed_const_iterator operator++(int) { auto x = *this; return ++i_, x; }
    indexed_const_iterator operator--(int) { auto x = *this; return --i_, x; }

    indexed_const_iterator operator+(difference_type n) { return indexed_const_iterator{ptr_, i_+n}; }
    indexed_const_iterator operator-(difference_type n) { return indexed_const_iterator{ptr_, i_-n}; }
    indexed_const_iterator& operator+=(difference_type n) { return i_+=n, *this; }
    indexed_const_iterator& operator-=(difference_type n) { return i_-=n, *this; }

    friend indexed_const_iterator operator+(difference_type n, indexed_const_iterator x) {
        indexed_const_iterator r(std::move(x));
        return r+=n;
    }

    reference operator*() const { return (*ptr_)[i_]; }
    pointer operator->() const { return pointer{(*ptr_)[i_]}; }
};


template <typename X = void>
struct pw_elements {
    using size_type = pw_size_type;
    static constexpr size_type npos = pw_npos;

    struct value_type {
        std::pair<double, double> interval;
        X element;

        bool operator==(const value_type& other) const { return interval==other.interval && element==other.element; }
        bool operator!=(const value_type& other) const { return interval!=other.interval || element!=other.element; }
    };

    using const_iterator = indexed_const_iterator<pw_elements<X>>;
    using iterator = const_iterator;

    // Consistency requirements:
    // 1. empty() || element_.size()+1 = vertex_.size()
    // 2. vertex_[i]<=vertex_[j] for i<=j.

    std::vector<double> vertex_;
    std::vector<X> element_;

    // Ctors and assignment:

    pw_elements() = default;

    template <typename VSeq, typename ESeq>
    pw_elements(const VSeq& vs, const ESeq& es) {
        assign(vs, es);
    }

    pw_elements(std::initializer_list<double> vs, std::initializer_list<X> es) {
        assign(vs, es);
    }

    pw_elements(pw_elements&&) = default;
    pw_elements(const pw_elements&) = default;

    template <typename Y>
    explicit pw_elements(const pw_elements<Y>& from):
        vertex_(from.vertex_), element_(from.element_.begin(), from.element_.end())
    {}

    pw_elements& operator=(pw_elements&&) = default;
    pw_elements& operator=(const pw_elements&) = default;

    // Access:

    auto intervals() const { return util::partition_view(vertex_); }
    auto interval(size_type i) const { return intervals()[i]; }

    auto bounds() const { return intervals().bounds(); }

    size_type size() const { return element_.size(); }
    bool empty() const { return size()==0; }

    bool operator==(const pw_elements& x) const {
        return vertex_==x.vertex_ && element_==x.element_;
    }

    bool operator!=(const pw_elements& x) const { return !(*this==x); }

    const auto& elements() const { return element_; }
    const auto& vertices() const { return vertex_; }

    X& element(size_type i) & { return element_[i]; }
    const X& element(size_type i) const & { return element_[i]; }
    value_type operator[](size_type i) const { return value_type{interval(i), element(i)}; }

    const_iterator cbegin() const { return const_iterator{this, 0}; }
    const_iterator begin() const { return cbegin(); }
    const_iterator cend() const { return const_iterator{this, size()}; }
    const_iterator end() const { return cend(); }
    value_type front() const { return (*this)[0]; }
    value_type back() const { return (*this)[size()-1]; }

    // Return index of right-most element whose corresponding closed interval contains x.
    size_type index_of(double x) const {
        if (empty()) return npos;

        auto partn = intervals();
        if (x == partn.bounds().second) return size()-1;
        else return partn.index(x);
    }

    // Return iterator pair spanning elements whose corresponding closed intervals contain x.
    std::pair<iterator, iterator> equal_range(double x) const {
        auto eq = std::equal_range(vertex_.begin(), vertex_.end(), x);

        // Let n be the number of elements, indexed from 0 to n-1, with
        // vertices indexed from 0 to n. Observe:
        // * eq.first points to least vertex v_i ≥ x.
        // * eq.second points to vertex_.end() if the last vertex v_n ≤ x,
        //   or else to the least vertex v_k > x.
        //
        // Elements then correspond to the index range [b, e), where:
        // * b=0 if i=0, else b=i-1, as v_i will be the upper vertex for
        //   the first element whose (closed) support contains x.
        // * e=k if k<n, since v_k will be the upper vertex for the
        //   the last element (index k-1) whose support contains x.
        //   Otherwise, if k==n or eq.second is vertex_.end(), the
        //   last element (index n-1) contains x, and so e=n.

        if (eq.first==vertex_.end()) return {end(), end()};
        if (eq.first>vertex_.begin()) --eq.first;
        if (eq.second==vertex_.end()) --eq.second;

        return {begin()+(eq.first-vertex_.begin()), begin()+(eq.second-vertex_.begin())};
    }

    value_type operator()(double x) const {
        size_type i = index_of(x);
        if (i==npos) {
            throw std::range_error("position outside support");
        }
        return (*this)[i];
    }

    // mutating operations:

    void reserve(size_type n) {
        vertex_.reserve(n+1);
        element_.reserve(n);
    }

    void clear() {
        vertex_.clear();
        element_.clear();
    }

    template <typename U>
    void push_back(double left, double right, U&& elem) {
        if (!empty() && left!=vertex_.back()) {
            throw std::runtime_error("noncontiguous element");
        }

        if (right<left) {
            throw std::runtime_error("inverted element");
        }

        // Extend element_ first in case a conversion/copy/move throws.

        element_.push_back(std::forward<U>(elem));
        if (vertex_.empty()) vertex_.push_back(left);
        vertex_.push_back(right);
    }

    template <typename U>
    void push_back(double right, U&& elem) {
        if (empty()) {
            throw std::runtime_error("require initial left vertex for element");
        }

        push_back(vertex_.back(), right, elem);
    }

    void assign(std::initializer_list<double> vs, std::initializer_list<X> es) {
        using util::make_range;
        assign(make_range(vs.begin(), vs.end()), make_range(es.begin(), es.end()));
    }

    template <typename Seq1, typename Seq2>
    void assign(const Seq1& vertices, const Seq2& elements) {
        using std::begin;
        using std::end;

        auto vi = begin(vertices);
        auto ve = end(vertices);

        auto ei = begin(elements);
        auto ee = end(elements);

        if (ei==ee) { // empty case
            if (vi!=ve) {
                throw std::runtime_error("vertex list too long");
            }
            clear();
            return;
        }

        double left = *vi++;
        if (vi==ve) {
            throw std::runtime_error("vertex list too short");
        }

        clear();

        double right = *vi++;
        push_back(left, right, *ei++);

        while (ei!=ee) {
            if (vi==ve) {
                throw std::runtime_error("vertex list too short");
            }
            double right = *vi++;
            push_back(right, *ei++);
        }

        if (vi!=ve) {
            throw std::runtime_error("vertex list too long");
        }
    }
};

// With X = void, present the element intervals only,
// keeping othewise the same interface.

template <> struct pw_elements<void> {
    using size_type = pw_size_type;
    static constexpr size_type npos = pw_npos;

    std::vector<double> vertex_;

    struct value_type {
        std::pair<double, double> interval;

        bool operator==(const value_type& other) const { return interval==other.interval; }
        bool operator!=(const value_type& other) const { return interval!=other.interval; }
    };

    using const_iterator = indexed_const_iterator<pw_elements<void>>;
    using iterator = const_iterator;

    // ctors and assignment:

    template <typename VSeq>
    explicit pw_elements(const VSeq& vs) { assign(vs); }

    pw_elements(std::initializer_list<double> vs) { assign(vs); }

    pw_elements() = default;
    pw_elements(pw_elements&&) = default;
    pw_elements(const pw_elements&) = default;

    template <typename Y>
    explicit pw_elements(const pw_elements<Y>& from):
        vertex_(from.vertex_) {}

    pw_elements& operator=(pw_elements&&) = default;
    pw_elements& operator=(const pw_elements&) = default;

    // access:

    auto intervals() const { return util::partition_view(vertex_); }
    auto interval(size_type i) const { return intervals()[i]; }
    value_type operator[](size_type i) const { return value_type{interval(i)}; }

    auto bounds() const { return intervals().bounds(); }

    size_type size() const { return vertex_.empty()? 0: vertex_.size()-1; }
    bool empty() const { return vertex_.empty(); }

    bool operator==(const pw_elements& x) const { return vertex_==x.vertex_; }
    bool operator!=(const pw_elements& x) const { return !(*this==x); }

    const auto& vertices() const { return vertex_; }

    size_type index_of(double x) const {
        if (empty()) return npos;

        auto partn = intervals();
        if (x == partn.bounds().second) return size()-1;
        else return partn.index(x);
    }

    // Return iterator pair spanning elements whose corresponding closed intervals contain x.
    std::pair<iterator, iterator> equal_range(double x) const {
        auto eq = std::equal_range(vertex_.begin(), vertex_.end(), x);

        if (eq.first==vertex_.end()) return {end(), end()};
        if (eq.first>vertex_.begin()) --eq.first;
        if (eq.second==vertex_.end()) --eq.second;

        return {begin()+(eq.first-vertex_.begin()), begin()+(eq.second-vertex_.begin())};
    }

    const_iterator cbegin() const { return const_iterator{this, 0}; }
    const_iterator begin() const { return cbegin(); }
    const_iterator cend() const { return const_iterator{this, size()}; }
    const_iterator end() const { return cend(); }
    value_type front() const { return (*this)[0]; }
    value_type back() const { return (*this)[size()-1]; }

    // mutating operations:

    void reserve(size_type n) { vertex_.reserve(n+1); }
    void clear() { vertex_.clear(); }

    void push_back(double left, double right) {
        if (!empty() && left!=vertex_.back()) {
            throw std::runtime_error("noncontiguous element");
        }

        if (right<left) {
            throw std::runtime_error("inverted element");
        }

        if (vertex_.empty()) vertex_.push_back(left);
        vertex_.push_back(right);
    }

    void push_back(double right) {
        if (empty()) {
            throw std::runtime_error("require initial left vertex for element");
        }
        vertex_.push_back(right);
    }

    void assign(std::initializer_list<double> vs) {
        assign(util::make_range(vs.begin(), vs.end()));
    }

    template <typename Seq1>
    void assign(const Seq1& vertices) {
        using std::begin;
        using std::end;

        auto vi = begin(vertices);
        auto ve = end(vertices);

        if (vi==ve) {
            clear();
            return;
        }

        double left = *vi++;
        if (vi==ve) {
            throw std::runtime_error("vertex list too short");
        }

        clear();

        double right = *vi++;
        push_back(left, right);

        while (vi!=ve) {
            double right = *vi++;
            push_back(right);
        }
    }
};

template <typename X>
using pw_element = typename pw_elements<X>::value_type;

namespace impl {
    template <typename A, typename B>
    struct piecewise_pairify {
        std::pair<A, B> operator()(
            double, double,
            const pw_element<A>& a_elem,
            const pw_element<B>& b_elem) const
         {
            return {a_elem.element, b_elem.element};
        }
    };

    template <typename X>
    struct piecewise_pairify<X, void> {
        X operator()(
            double, double,
            const pw_element<X>& a_elem,
            const pw_element<void>& b_elem) const
        {
            return a_elem.element;
        }
    };

    template <typename X>
    struct piecewise_pairify<void, X> {
        X operator()(
            double, double,
            const pw_element<void>& a_elem,
            const pw_element<X>& b_elem) const
        {
            return b_elem.element;
        }
    };

    template <>
    struct piecewise_pairify<void, void> {
        void operator()(
            double, double,
            const pw_element<void>&,
            const pw_element<void>&) const {}
    };
}

// Zip combines successive elements from two pw_elements sequences where the
// elements overlap. Let A_i and B_i denote the ordered elements from each of
// the sequences A and B, and Z_i denote the elements from the resulting
// sequence Z.
//
// * The support (`bounds()`) of the zipped result is the intersections of the
//   support of the two sequences.
//
// * Each element Z_k in the zip corresponds to the intersection of some
//   element A_i and B_j. The extent of Z_k is the intersection of the
//   extents of A_i and B_j, and its value is determined by the supplied
//   `combine` function.
//
// * For every element in A_i in A, if A_i intersects with an element of
//   B, then there will be an Z_k corresponding to the intersection of A_i
//   with some element of B. Likewise for elements B_i of B.
//
// * Elements of Z respect the ordering of elements in A and B, and do
//   not repeat. If Z_k is derived from A_i and B_j, then Z_(k+1) is derived
//   from A_(i+1) and B_(j+1) if possible, or else from A_i and B_(j+1) or
//   A_(i+1) and B_j.
//
// The Combine functional takes four arguments: double left, double right,
// pw_elements<A>::value_type, pw_elements<B>::value_type b. The default
// combine functional returns std::pair<A, B>, unless one of A and B is void.
//
// TODO: Consider making a lazy `zip_view` version of zip.

template <typename A, typename B, typename Combine = impl::piecewise_pairify<A, B>>
auto zip(const pw_elements<A>& a, const pw_elements<B>& b, Combine combine = {})
{
    using Out = decltype(combine(0., 0., a.front(), b.front()));
    constexpr bool is_void = std::is_void_v<Out>;

    pw_elements<Out> z;
    if (a.empty() || b.empty()) return z;

    double lmax = std::max(a.bounds().first, b.bounds().first);
    double rmin = std::min(a.bounds().second, b.bounds().second);
    if (rmin<lmax) return z;

    auto ai = a.equal_range(lmax).first;
    auto bi = b.equal_range(lmax).first;

    auto a_end = a.equal_range(rmin).second;
    auto b_end = b.equal_range(rmin).second;

    double left = lmax;
    double a_right = ai->interval.second;
    double b_right = bi->interval.second;
    for (;;) {
        double right = std::min(a_right, b_right);
        if constexpr (is_void) {
            z.push_back(left, right);
        }
        else {
            z.push_back(left, right, combine(left, right, *ai, *bi));
        }

        bool advance_a = a_right==right && std::next(ai)!=a_end;
        bool advance_b = b_right==right && std::next(bi)!=b_end;
        if (!advance_a && !advance_b) break;

        if (advance_a) a_right = (++ai)->interval.second;
        if (advance_b) b_right = (++bi)->interval.second;
        left = right;
    }
    return z;
}

} // namespace util
} // namespace arb
