#pragma once

// Create/manipulate 1-d piecewise defined objects.
//
// A `pw_element<A>` describes a _value_ of type `A` and an _extent_ of
// type std::pair<double, double>. An extent with value `{l, r}` has
// l ≤ r and represents a closed interval [l, r] of the real line.
//
// For void `A`, `pw_element<void>` holds only the extent of an element.
//
// Once constructed, the value of a `pw_element<A>` (with `A` not void) is
// mutable, and can be assigned directly with `operator=`, but the extent is
// constant. A `pw_element<A>` can be implicitly converted to a value of type
// `A`.
//
// A `pw_element<A>` can be bound via a structured binding to an extent
// and value, or just an extent if A is void.
//
// A `pw_elements<A>` object comprises a contiguous sequence Eᵢ of
// `pw_element<A>` elements. If Eᵢ has extent [lᵢ, rᵢ] and Eᵢ₊₁ has extent
// [lᵢ₊₁, rᵢ₊₁] then rᵢ must equal lᵢ₊₁.
//
// The `value_type` of `pw_elements<A>` is the type of the elements, i.e.
// `pw_element<A>`, as `pw_elements<A>` presents as a container.
// To avoid ambiguity the type `A` is termed the _codomain_ of a `pw_elements<A>`
// object.
//
// When the codomain is `void`, which is also the default for the type
// parameter `A`, `pw_element<>` contains only the extent, and `pw_elements<>`
// does not hold any values for the elements.
//
//
// Construction:
//
// A `pw_elements<A>` object can be constructed from a sequence of _vertices_
// and a sequence of _values_ (if `A` is not `void`). A vertex sequence of
// doubles x₀, x₁, …, xₙ and value sequence v₀, v₁, …, vₙ₋₁ defines n elements
// with extents [x₀, x₁], [x₁, x₂], etc.
//
// A default constructed `pw_elements<A>` has no elements.
//
// A `pw_element<A>` can be appended to a `pw_elements<A>` object with:
//
//     pw_elements<A>::push_back(pw_element<A>)
//
// The lower bound of the element must equal the upper bound of the existing
// sequence, if it is not empty.
//
// The extent and value can be given explicitly with:
//
//     template <typename U&&>
//     pw_elements<A>::push_back(double left, double right, U&& value)
//
//     template <typename U&&>
//     pw_elements<A>::push_back(double right, U&& value)
//
// where the lower bound of the extent of the new element can be omitted
// if the sequence is non-empty. When `A` is void, these methods are
// instead:
//
//     pw_elements<>::push_back(double left, double right)
//     pw_elements<>::push_back(double right)
//
//
// Conversion:
//
// A `pw_elements<A>` object can be explicitly constructed from a `pw_elements<B>`
// object if the values of type B can be converted to values of type A.
//
// A `pw_elements<void>` object can be explicitly constucted from any
// `pw_elements<B>` object, where it takes just the element extents.
//
//
// Element access:
//
// Elements are retrievable by index by `operator[]`. The value of the ith
// element is given by `value(i)`, and its extent as a pair of doubles by
// `extent(i)`.
//
// The method `vertices()` returns the vector of nodes defining the element
// extents, while for non-void codomains, `values()` returns the vector of
// element values.
//
// Elements can be looked up by their position with `operator()(double x)` and
// `equal_range(double x)`. The element (or element proxy, see below) retrieved
// for `x` is the right-most element whose extent includes `x`; this makes
// `pw_elements<A>` act like a right-continuous function of position.
// `equal_range(x)` returns a pair of iterators that correspond to the sequence
// of elements whose extents all include `x`.
//
// Elements that are returned by a non-const lvalue `pw_elements<A>` via
// `operator[]`, `operator()`, or via an iterator are represented by a proxy
// object that allows in-place mutation of the element's value.

#include <cmath>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <vector>

#include "util/iterutil.hpp"
#include "util/transform.hpp"
#include "util/meta.hpp"
#include "util/partition.hpp"

namespace arb {
namespace util {

using pw_size_type = unsigned;
constexpr pw_size_type pw_npos = -1;

template <typename X = void>
struct pw_element {
    pw_element():
        extent(NAN, NAN),
        value()
    {}

    pw_element(std::pair<double, double> extent, X value):
        extent(std::move(extent)),
        value(std::move(value))
    {}

    pw_element(const pw_element&) = default;
    pw_element(pw_element&&) = default;

    operator X() const { return value; }
    pw_element& operator=(X x) { value = std::move(x); return *this; };

    double lower_bound() const { return extent.first; }
    double upper_bound() const { return extent.second; }

    const std::pair<double, double> extent;
    X value;
};

template <>
struct pw_element<void> {
    pw_element():
        extent(NAN, NAN)
    {}

    pw_element(std::pair<double, double> extent):
        extent(std::move(extent))
    {}

    pw_element(const pw_element&) = default;
    pw_element(pw_element&&) = default;

    double lower_bound() const { return extent.first; }
    double upper_bound() const { return extent.second; }

    const std::pair<double, double> extent;
};

template <typename X = void>
struct pw_elements;

// Proxy object for mutable iterators into a pw_elements<X> object.
template <typename X>
struct pw_element_proxy {
    pw_element_proxy(pw_elements<X>& pw, pw_size_type i):
        extent(pw.extent(i)), value(pw.value(i)) {}

    operator pw_element<X>() const { return pw_element<X>{extent, value}; }
    operator X() const { return value; }
    pw_element_proxy& operator=(X x) {value = std::move(x); return *this; };

    double lower_bound() const { return extent.first; }
    double upper_bound() const { return extent.second; }

    const std::pair<double, double> extent;
    X& value;
};

// Compute indices into vertex set corresponding to elements that cover a point x:

namespace {
std::pair<std::ptrdiff_t, std::ptrdiff_t> equal_range_indices(const std::vector<double>& vertices, double x) {
    if (vertices.empty()) return {0, 0};

    auto eq = std::equal_range(vertices.begin(), vertices.end(), x);

    // Let n be the number of elements, indexed from 0 to n-1, with
    // vertices indexed from 0 to n. Observe:
    // * eq.first points to least vertex v_i ≥ x.
    //   or else to vertices.end() if v < x for all vertices v.
    // * eq.second points to vertices.end() if the last vertex v_n ≤ x,
    //   or else to the least vertex v_k > x.
    //
    // Elements then correspond to the index range [b, e), where:
    // * b=0 if i=0, else b=i-1, as v_i will be the upper vertex for
    //   the first element whose (closed) support contains x.
    // * e=k if k<n, since v_k will be the upper vertex for the
    //   the last element (index k-1) whose support contains x.
    //   Otherwise, if k==n or eq.second is vertices.end(), the
    //   last element (index n-1) contains x, and so e=n.

    if (eq.first==vertices.end()) return {0, 0};
    if (eq.first>vertices.begin()) --eq.first;
    if (eq.second==vertices.end()) --eq.second;

    return std::make_pair(eq.first-vertices.begin(), eq.second-vertices.begin());
}
} // anonymous namespace

template <typename X>
struct pw_elements {
    using size_type = pw_size_type;
    using difference_type = std::make_signed_t<pw_size_type>;
    static constexpr size_type npos = pw_npos;

    using value_type = pw_element<X>;
    using codomain = X;

    struct iterator: iterator_adaptor<iterator, counter<pw_size_type>> {
        using typename iterator_adaptor<iterator, counter<pw_size_type>>::difference_type;
        iterator(pw_elements<X>& pw, pw_size_type index): pw_(&pw), c_(index) {}
        iterator(): pw_(nullptr) {}

        using value_type = pw_element<X>;
        using pointer = pointer_proxy<pw_element_proxy<X>>;
        using reference = pw_element_proxy<X>;

        reference operator[](difference_type j) { return reference{*pw_, j+*c_}; }
        reference operator*() { return reference{*pw_, *c_}; }
        value_type operator*() const { return (*pw_)[*c_]; }
        pointer operator->() { return pointer{*pw_, *c_}; }

        // (required for iterator_adaptor)
        counter<pw_size_type>& inner() { return c_; }
        const counter<pw_size_type>& inner() const { return c_; }

    protected:
        pw_elements<X>* pw_;
        counter<pw_size_type> c_;
    };

    struct const_iterator: iterator_adaptor<const_iterator, counter<pw_size_type>> {
        using typename iterator_adaptor<const_iterator, counter<pw_size_type>>::difference_type;
        const_iterator(const pw_elements<X>& pw, pw_size_type index): pw_(&pw), c_(index) {}
        const_iterator(): pw_(nullptr) {}

        using value_type = pw_element<X>;
        using pointer = const pointer_proxy<pw_element<X>>;
        using reference = pw_element<X>;

        reference operator[](difference_type j) const { return (*pw_)[j+*c_]; }
        reference operator*() const { return (*pw_)[*c_]; }
        pointer operator->() const { return pointer{(*pw_)[*c_]}; }

        // (required for iterator_adaptor)
        counter<pw_size_type>& inner() { return c_; }
        const counter<pw_size_type>& inner() const { return c_; }

    protected:
        const pw_elements<X>* pw_;
        counter<pw_size_type> c_;
    };

    // Ctors and assignment:

    pw_elements() = default;

    template <typename VertexSeq, typename ValueSeq>
    pw_elements(const VertexSeq& vs, const ValueSeq& es) {
        assign(vs, es);
    }

    pw_elements(std::initializer_list<double> vs, std::initializer_list<X> es) {
        assign(vs, es);
    }

    pw_elements(pw_elements&&) = default;
    pw_elements(const pw_elements&) = default;

    template <typename Y>
    explicit pw_elements(const pw_elements<Y>& from):
        vertex_(from.vertex_), value_(from.value_.begin(), from.value_.end())
    {}

    pw_elements& operator=(pw_elements&&) = default;
    pw_elements& operator=(const pw_elements&) = default;

    // Access:

    auto extents() const { return util::partition_view(vertex_); }

    auto bounds() const { return extents().bounds(); }
    auto lower_bound() const { return bounds().first; }
    auto upper_bound() const { return bounds().second; }

    auto extent(size_type i) const { return extents()[i]; }
    auto lower_bound(size_type i) const { return extents()[i].first; }
    auto upper_bound(size_type i) const { return extents()[i].second; }

    size_type size() const { return value_.size(); }
    bool empty() const { return size()==0; }

    bool operator==(const pw_elements& x) const {
        return vertex_==x.vertex_ && value_==x.value_;
    }

    bool operator!=(const pw_elements& x) const { return !(*this==x); }

    const auto& values() const { return value_; }
    const auto& vertices() const { return vertex_; }

    X& value(size_type i) & { return value_[i]; }
    const X& value(size_type i) const & { return value_[i]; }
    X value(size_type i) const && { return value_[i]; }

    auto operator[](size_type i) & { return pw_element_proxy<X>{*this, i}; }
    auto operator[](size_type i) const & { return value_type{extent(i), value(i)}; }
    auto operator[](size_type i) const && { return value_type{extent(i), value(i)}; }

    auto front() & { return pw_element_proxy<X>{*this, 0}; }
    auto front() const & { return value_type{extent(0), value(0)}; }
    auto front() const && { return value_type{extent(0), value(0)}; }

    auto back() & { return pw_element_proxy<X>{*this, size()-1}; }
    auto back() const & { return value_type{extent(size()-1), value(size()-1)}; }
    auto back() const && { return value_type{extent(size()-1), value(size()-1)}; }

    iterator begin() { return iterator{*this, 0}; }
    iterator end() { return iterator{*this, size()}; }

    const_iterator cbegin() const { return const_iterator{*this, 0}; }
    const_iterator begin() const { return cbegin(); }
    const_iterator cend() const { return const_iterator{*this, size()}; }
    const_iterator end() const { return cend(); }

    // Return index of right-most element whose corresponding closed interval contains x.
    size_type index_of(double x) const {
        if (empty()) return npos;

        auto partn = extents();
        if (x == partn.bounds().second) return size()-1;
        else return partn.index(x);
    }

    // Return iterator pair spanning elements whose corresponding closed extents contain x.
    std::pair<iterator, iterator> equal_range(double x) {
        auto qi = equal_range_indices(vertex_, x);
        return {begin()+qi.first, begin()+qi.second};
    }

    std::pair<const_iterator, const_iterator> equal_range(double x) const {
        auto qi = equal_range_indices(vertex_, x);
        return {cbegin()+qi.first, cbegin()+qi.second};
    }

    auto operator()(double x) const && {
        size_type i = index_of(x);
        return i!=npos? (*this)[i]: throw std::range_error("position outside support");
    }

    auto operator()(double x) & {
        size_type i = index_of(x);
        return i!=npos? (*this)[i]: throw std::range_error("position outside support");
    }

    auto operator()(double x) const & {
        size_type i = index_of(x);
        return i!=npos? (*this)[i]: throw std::range_error("position outside support");
    }

    // mutating operations:

    void reserve(size_type n) {
        vertex_.reserve(n+1);
        value_.reserve(n);
    }

    void clear() {
        vertex_.clear();
        value_.clear();
    }

    void push_back(pw_element<X> elem) {
        double left = elem.lower_bound();
        double right = elem.upper_bound();
        push_back(left, right, std::move(elem.value));
    }

    template <typename U>
    void push_back(double left, double right, U&& v) {
        if (!empty() && left!=vertex_.back()) {
            throw std::runtime_error("noncontiguous element");
        }

        if (right<left) {
            throw std::runtime_error("inverted element");
        }

        // Extend value_ first in case a conversion/copy/move throws.
        value_.push_back(std::forward<U>(v));
        if (vertex_.empty()) vertex_.push_back(left);
        vertex_.push_back(right);
    }

    template <typename U>
    void push_back(double right, U&& v) {
        if (empty()) {
            throw std::runtime_error("require initial left vertex for element");
        }

        push_back(vertex_.back(), right, std::forward<U>(v));
    }

    void assign(std::initializer_list<double> vs, std::initializer_list<X> es) {
        using util::make_range;
        assign(make_range(vs.begin(), vs.end()), make_range(es.begin(), es.end()));
    }

    template <typename Seq1, typename Seq2>
    void assign(const Seq1& vertices, const Seq2& values) {
        using std::begin;
        using std::end;

        auto vi = begin(vertices);
        auto ve = end(vertices);

        auto ei = begin(values);
        auto ee = end(values);

        if (ei==ee) { // empty case
            if (vi!=ve) {
                throw std::runtime_error("vertex list too long");
            }
            clear();
            return;
        }

        if (vi==ve) {
            throw std::runtime_error("vertex list too short");
        }

        clear();

        double left = *vi++;
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

private:
    // Consistency requirements:
    // 1. empty() || value_.size()+1 = vertex_.size()
    // 2. vertex_[i]<=vertex_[j] for i<=j.

    std::vector<double> vertex_;
    std::vector<X> value_;
};

// With X = void, present the element intervals only, keeping otherwise the
// same interface.

template <>
struct pw_elements<void> {
    using size_type = pw_size_type;
    static constexpr size_type npos = pw_npos;
    using difference_type = std::make_signed_t<pw_size_type>;

    using value_type = pw_element<void>;
    using codomain = void;

    struct const_iterator: iterator_adaptor<const_iterator, counter<pw_size_type>> {
        using typename iterator_adaptor<const_iterator, counter<pw_size_type>>::difference_type;
        const_iterator(const pw_elements<void>& pw, pw_size_type index): pw_(&pw), c_(index) {}
        const_iterator(): pw_(nullptr) {}

        using value_type = pw_element<void>;
        using pointer = const pointer_proxy<pw_element<void>>;
        using reference = pw_element<void>;

        reference operator[](difference_type j) const { return (*pw_)[j+*c_]; }
        reference operator*() const { return (*pw_)[*c_]; }
        pointer operator->() const { return pointer{(*pw_)[*c_]}; }

        // (required for iterator_adaptor)
        counter<pw_size_type>& inner() { return c_; }
        const counter<pw_size_type>& inner() const { return c_; }

    protected:
        const pw_elements<void>* pw_;
        counter<pw_size_type> c_;
    };

    using iterator = const_iterator;

    // Ctors and assignment:

    pw_elements() = default;

    template <typename VertexSeq>
    pw_elements(const VertexSeq& vs) {
        assign(vs);
    }

    pw_elements(std::initializer_list<double> vs) {
        assign(vs);
    }

    pw_elements(pw_elements&&) = default;
    pw_elements(const pw_elements&) = default;

    template <typename Y>
    explicit pw_elements(const pw_elements<Y>& from):
        vertex_(from.vertices()) {}

    pw_elements& operator=(pw_elements&&) = default;
    pw_elements& operator=(const pw_elements&) = default;

    // Access:

    auto extents() const { return util::partition_view(vertex_); }

    auto bounds() const { return extents().bounds(); }
    auto lower_bound() const { return bounds().first; }
    auto upper_bound() const { return bounds().second; }

    auto extent(size_type i) const { return extents()[i]; }
    auto lower_bound(size_type i) const { return extents()[i].first; }
    auto upper_bound(size_type i) const { return extents()[i].second; }

    size_type size() const { return empty()? 0: vertex_.size()-1; }
    bool empty() const { return vertex_.empty(); }

    bool operator==(const pw_elements& x) const {
        return vertex_==x.vertex_;
    }

    bool operator!=(const pw_elements& x) const { return !(*this==x); }

    const auto& vertices() const { return vertex_; }

    void value(size_type i) const {}
    pw_element<void> operator[](size_type i) const { return value_type{extent(i)}; }

    pw_element<void> front() const { return value_type{extent(0)}; }
    pw_element<void> back() const { return value_type{extent(size()-1)}; }

    const_iterator cbegin() const { return const_iterator{*this, 0}; }
    const_iterator begin() const { return cbegin(); }
    const_iterator cend() const { return const_iterator{*this, size()}; }
    const_iterator end() const { return cend(); }

    // Return index of right-most element whose corresponding closed interval contains x.
    size_type index_of(double x) const {
        if (empty()) return npos;

        auto partn = extents();
        if (x == partn.bounds().second) return size()-1;
        else return partn.index(x);
    }

    // Return iterator pair spanning elements whose corresponding closed extents contain x.
    std::pair<const_iterator,const_iterator> equal_range(double x) const {
        auto qi = equal_range_indices(vertex_, x);
        return {cbegin()+qi.first, cbegin()+qi.second};
    }

    auto operator()(double x) const {
        size_type i = index_of(x);
        return i!=npos? (*this)[i]: throw std::range_error("position outside support");
    }

    // mutating operations:

    void reserve(size_type n) {
        vertex_.reserve(n+1);
    }

    void clear() {
        vertex_.clear();
    }

    void push_back(const pw_element<void>& elem) {
        double left = elem.lower_bound();
        double right = elem.upper_bound();
        push_back(left, right);
    }

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

        push_back(vertex_.back(), right);
    }

    void assign(std::initializer_list<double> vs) {
        assign(make_range(vs.begin(), vs.end()));
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

private:
    // Consistency requirements:
    // 1. empty() || value_.size()+1 = vertex_.size()
    // 2. vertex_[i]<=vertex_[j] for i<=j.

    std::vector<double> vertex_;
};

// The piecewise map applies a transform to the values in a piecewise
// object, preserving the extents. If the original piecewise object
// is `pw_elements<void>`, then the function is called with zero arguments.
//
// If the mapping function returns void, return a `pw_elements<void>` object
// with the same extents as the original piecewise object.

template <typename X, typename Fn>
auto pw_map(const pw_elements<X>& pw, Fn&& fn) {
    if constexpr (std::is_void_v<X>) {
        using Out = std::invoke_result_t<Fn&&>;
        if constexpr (std::is_void_v<Out>) {
            // Evalate fn for side effects.
            for (const auto& elem: pw) fn();
            return pw;
        }
        else {
            auto mapped = util::transform_view(pw, [&fn](auto&&) { return fn(); });
            return pw_elements<Out>(pw.vertices(), std::vector<Out>(mapped.begin(), mapped.end()));
        }
    }
    else {
        using Out = std::invoke_result_t<Fn&&, X>;
        if constexpr (std::is_void_v<Out>) {
            // Evalate fn for side effects.
            for (const auto& v: pw.values()) fn(v);
            return pw_elements<void>(pw);
        }
        else {
            auto mapped = util::transform_view(pw, [&fn](auto&& elem) { return fn(elem.value); });
            return pw_elements<Out>(pw.vertices(), std::vector<Out>(mapped.begin(), mapped.end()));
        }
    }
}

// The piecewise zip combines successive elements from two `pw_elements`
// sequences where the elements overlap.
//
// * `pw_zip_view` performs a lazy zip, represented by a range of
//   `pw_zip_iterator` objects. The iterators dereference to objects of
//   type `pw_element<std::pair<pw_element<A>, pw_element<B>>`.
//
// * `pw_zip` performs a strict zip, returning a piecewise object of type
//   `pw_elements<std::pair<pw_element<A>, pw_element<B>>`.
//
// * `pw_zip_with` performs a map composed with a zip, where the map is
//    given by a function which takes three arguments: the extent of
//    the zipped element as `std::pair<double, double>`; the element
//    from the first sequence of type `<pw_element<A>`; and the element
//    from the second sequence of type `pw_element<B>`. It is equivalent
//    in action to performing a `pw_zip` followed by a `pw_map`.
//
//    By default, `pw_zip_with` uses `pw_pairify` to combine elements,
//    which returns a tuple of the non-void values from each pair of elements
//    in the zip.
//
// The elements of the zip are determined as follows:
//
// Let A_i and B_i denote the ordered elements from each of the sequences A and
// B, and Z_i denote the elements forming the resulting sequence Z.
//
// * Each element Z_k in the zip corresponds to the intersection of an
//   element A_i and B_j. The extent of Z_k is the intersection of the
//   extents of A_i and B_j, and its value is the pair {A_i, B_j}.
//
// * For every element in A_i in A, if A_i intersects with an element of
//   B, then there will be an Z_k corresponding to the intersection of A_i
//   with some element of B. Likewise for elements B_i of B.
//
// * Elements of Z respect the ordering of elements in A and B, and do
//   not repeat. If Z_k is derived from A_i and B_j, then Z_(k+1) is derived
//   from A_(i+1) and B_(j+1) if possible, or else from A_i and B_(j+1) or
//   A_(i+1) and B_j.

template <typename A, typename B>
struct pw_zip_iterator {
    using value_type = pw_element<std::pair<pw_element<A>, pw_element<B>>>;
    using pointer = pointer_proxy<value_type>;
    using reference = value_type;
    using difference_type = std::ptrdiff_t;

    // Proxy iterator so not _really_ a forward iterator.
    using iterator_category = std::forward_iterator_tag;

    bool is_end = true;
    typename pw_elements<A>::const_iterator ai, a_end;
    typename pw_elements<B>::const_iterator bi, b_end;
    double left = std::nan("");

    pw_zip_iterator() = default;
    pw_zip_iterator(const pw_elements<A>& a, const pw_elements<B>& b) {
        double lmax = std::max(a.lower_bound(), b.lower_bound());
        double rmin = std::min(a.upper_bound(), b.upper_bound());

        is_end = rmin<lmax;
        if (!is_end) {
            ai = a.equal_range(lmax).first;
            a_end = a.equal_range(rmin).second;
            bi = b.equal_range(lmax).first;
            b_end = b.equal_range(rmin).second;
            left = lmax;
        }
    }

    pw_zip_iterator(const pw_zip_iterator& other):
        is_end(other.is_end),
        ai(other.ai),
        a_end(other.a_end),
        bi(other.bi),
        b_end(other.b_end),
        left(other.left)
    {}

    bool operator==(pw_zip_iterator other) const {
        if (is_end && other.is_end) return true;
        return is_end==other.is_end && ai==other.ai && a_end==other.a_end && bi==other.bi && b_end==other.b_end && left==other.left;
    }

    bool operator!=(pw_zip_iterator other) const {
        return !(*this==other);
    }

    pw_zip_iterator& operator++() {
        if (is_end) return *this;

        double a_right = ai->upper_bound();
        double b_right = bi->upper_bound();
        double right = std::min(a_right, b_right);

        bool advance_a = a_right==right && std::next(ai)!=a_end;
        bool advance_b = b_right==right && std::next(bi)!=b_end;
        if (!advance_a && !advance_b) {
            is_end = true;
        }
        else {
            if (advance_a) ++ai;
            if (advance_b) ++bi;
            left = right;
        }

        return *this;
    }

    pw_zip_iterator operator++(int) {
        pw_zip_iterator here = *this;
        ++*this;
        return here;
    }

    value_type operator*() const {
        double a_right = ai->upper_bound();
        double b_right = bi->upper_bound();
        double right = std::min(a_right, b_right);

        return value_type{{left, right}, {*ai, *bi}};
    }

    pointer operator->() const {
        return pointer{*this};
    }
};

template <typename A, typename B>
range<pw_zip_iterator<A, B>> pw_zip_range(const pw_elements<A>& a, const pw_elements<B>& b) {
    return {pw_zip_iterator<A, B>{a, b}, pw_zip_iterator<A, B>{}};
}

template <typename A, typename B>
auto pw_zip(const pw_elements<A>& a, const pw_elements<B>& b) {
    pw_elements<std::pair<pw_element<A>, pw_element<B>>> out;
    for (auto elem: pw_zip_range(a, b)) {
        out.push_back(elem);
    }
    return out;
}

// `pw_pairify` is a functional intended for use with `pw_zip_with` that takes
// an extent, which is ignored, and two elements of types `pw_element<A>` and
// `pw_element<B>` and returns void if both A and B are void, or the pair of
// their values if neither A nor B is void, or else the single non-void value
// of type A or B.

namespace impl {
    template <typename A, typename B>
    std::pair<A, B> pw_pairify_(const pw_element<A>& a_elem, const pw_element<B>& b_elem) {
        return {a_elem.value, b_elem.value};
    };

    template <typename A>
    A pw_pairify_(const pw_element<A>& a_elem, const pw_element<void>&) {
        return a_elem.value;
    };

    template <typename B>
    B pw_pairify_(const pw_element<void>&, const pw_element<B>& b_elem) {
        return b_elem.value;
    };

    inline void pw_pairify_(const pw_element<void>&, const pw_element<void>&) {}
}

struct pw_pairify {
    template <typename A, typename B>
    auto operator()(std::pair<double, double>, const pw_element<A>& a_elem, const pw_element<B>& b_elem) const {
        return impl::pw_pairify_(a_elem, b_elem);
    }
};

template <typename A, typename B, typename Fn = pw_pairify>
auto pw_zip_with(const pw_elements<A>& a, const pw_elements<B>& b, Fn&& fn = Fn{}) {
    using Out = decltype(fn(std::pair<double, double>{}, a.front(), b.front()));
    pw_elements<Out> out;

    for (auto elem: pw_zip_range(a, b)) {
        if constexpr (std::is_void_v<Out>) {
            fn(elem.extent, elem.value.first, elem.value.second);
            out.push_back(elem.extent.first, elem.extent.second);
        }
        else {
            out.push_back(elem.extent.first, elem.extent.second,
                fn(elem.extent, elem.value.first, elem.value.second));
        }
    }
    return out;
}

} // namespace util
} // namespace arb
