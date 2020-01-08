#pragma once

// An element representing a segment of a rational polynomial function
// of order p, q, as determined by its values on n = p+q+1 nodes at
// [0, 1/n, ..., 1].
//
// Rational polynomial interpolation scheme from:
// F. M. Larkin (1967). Some techniques for rational interpolation.
// The Computer Journal 10(2), pp. 178–187.
//
// TODO: Consider implementing a more generally robust scheme, e.g.
// S. L. Loi and A. W. McInnes (1983). An algorithm for generalized
// rational interpolation. BIT Numerical Mathematics 23(1),
// pp. 105–117. doi:10.1007/BF01937330
//
// Current implementation should be sufficient for interpolation of
// monotonic ratpoly functions of low order, but we can revisit
// this is if the limitations of the Larkin method bite us.

#include <algorithm>
#include <array>
#include <functional>
#include <type_traits>
#include <utility>

#include <arbor/math.hpp>

namespace arb {
namespace util {

namespace impl {

template <unsigned n, unsigned sz>
struct array_init_n {
    template <typename A, typename X, typename... Tail>
    static void set(A& array, X value, Tail... tail) {
        array[sz-n] = std::move(value);
        array_init_n<n-1, sz>::set(array, std::forward<Tail>(tail)...);
    }
};

template <unsigned sz>
struct array_init_n<0, sz> {
    template <typename A>
    static void set(A& array) {}
};

// TODO: C++17. The following is much nicer with if constexpr...

template <unsigned a, unsigned c, unsigned k, bool upper>
struct rat_eval {
    static double eval(const std::array<double, 1+a+c>& g, double x) {
        std::array<double, a+c> h;
        interpolate(h, g, x);
        return rat_eval<a-1, c, k+1, upper>::eval(h, g, x);
    }

    static double eval(const std::array<double, 1+a+c>& g, const std::array<double, 2+a+c>&, double x) {
        std::array<double, a+c> h;
        interpolate(h, g, x);
        return rat_eval<a-1, c, k+1, upper>::eval(h, g, x);
    }

    static void interpolate(std::array<double, a+c>& h, const std::array<double, a+c+1>& g, double x) {
        constexpr double ook = 1./k;
        for (unsigned i = 0; i<a+c; ++i) {
            if (upper) {
                h[i] = ook*((x - i)*g[i+1] + (i+k - x)*g[i]);
            }
            else {
                // Using h[i] = k/(g[i+1]/(x - i) + g[i]/(i+k - x)) is more robust to
                // singularities, but the expense should not be necessary if we stick
                // to strictly monotonic elements for rational polynomials.
                h[i] = k*g[i]*g[i+1]/(g[i]*(x - i) + g[i+1]*(i+k - x));
            }
        }
    }
};

template <unsigned k, bool upper>
struct rat_eval<0, 0, k, upper> {
    static double eval(const std::array<double, 1>& g, double) {
        return g[0];
    }

    static double eval(const std::array<double, 1>& g, const std::array<double, 2>&, double) {
        return g[0];
    }
};

template <unsigned c, unsigned k, bool upper>
struct rat_eval<0, c, k, upper> {
    static double eval(const std::array<double, 1+c>& g, const std::array<double, 2+c>& p, double x) {
        // 'rhombus' interpolation
        std::array<double, c> h;
        for (unsigned i = 0; i<c; ++i) {
            h[i] = p[i+1] + k/((x - i)/(g[i+1]-p[i+1]) + (i+k - x)/(g[i]-p[i+1]));
        }

        return rat_eval<0, c-1, k+1, upper>::eval(h, g, x);
    }
};

} // namespace impl

template <unsigned p, unsigned q>
struct rat_element {
    // Construct from function evaluated on nodes.
    template <
        typename F,
        typename _ = std::enable_if_t<std::is_convertible<decltype(std::declval<F>()(0.0)), double>::value>
    >
    explicit rat_element(F&& fn, _* = nullptr) {
        if (size()>1) {
            for (unsigned i = 0; i<size(); ++i) data_[i] = fn(i/(size()-1.0));
        }
        else {
            data_[0] = fn(0);
        }
    }

    // Construct from node values.
    template <typename... Tail>
    rat_element(double y0, Tail... tail) {
        impl::array_init_n<p+q+1, p+q+1>::set(data_, y0, tail...);
    }

    // Construct from node values in array or std::array.
    template <typename Y>
    rat_element(const std::array<Y, 1+p+q>& a) { unchecked_range_init(a); }

    template <typename Y>
    rat_element(Y (&a)[1+p+q]) { unchecked_range_init(a); }

    // Number of nodes.
    static constexpr unsigned size() { return 1+p+q; }

    // Rational interpolation at x.
    double operator()(double x) const {
        // upper j => interpolate polynomials first;
        // !upper => interpolate reciprocal polynomials first;
        // a is the number of (reciprocal) polynomial interpolation steps;
        // c is the number of 'rhombus' interpolation steps.

        constexpr bool upper = p>=q;
        constexpr unsigned a = upper? p-q+(q>0): q-p+(p>0);
        constexpr unsigned c = p+q-a;

        return impl::rat_eval<a, c, 1, upper>::eval(data_, x*(p+q));
    }

    // Node values.
    double operator[](unsigned i) const {
        return data_.at(i);
    }

    double& operator[](unsigned i) {
        return data_.at(i);
    }

private:
    template <typename C>
    void unchecked_range_init(const C& a) {
        using std::begin;
        using std::end;
        std::copy(begin(a), end(a), data_.begin());
    }

    std::array<double, 1+p+q> data_;
};

} // namespace util
} // namespace arb
