#pragma once

/* Present an integral value as an iterator, for integral-range 'containers' */

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace arb {
namespace util {

template <typename V, typename = std::enable_if_t<std::is_integral<V>::value>>
struct counter {
    using difference_type = V;
    using value_type = V;
    using pointer = const V*;
    using reference = const V&;
    using iterator_category = std::random_access_iterator_tag;

    counter(): v_{} {}
    counter(V v): v_{v} {}

    counter(const counter&) = default;
    counter(counter&&) = default;

    counter& operator++() {
        ++v_;
        return *this;
    }

    counter operator++(int) {
        counter c(*this);
        ++v_;
        return c;
    }

    counter& operator--() {
        --v_;
        return *this;
    }

    counter operator--(int) {
        counter c(*this);
        --v_;
        return c;
    }

    counter& operator+=(difference_type n) {
        v_ += n;
        return *this;
    }

    counter& operator-=(difference_type n) {
        v_ -= n;
        return *this;
    }

    counter operator+(difference_type n) {
        return counter(v_+n);
    }

    friend counter operator+(difference_type n, counter x) {
        return counter(n+x.v_);
    }

    counter operator-(difference_type n) {
        return counter(v_-n);
    }

    difference_type operator-(counter x) const {
        return v_-x.v_;
    }

    value_type operator*() const { return v_; }

    pointer operator->() const { return &v_; }

    value_type operator[](difference_type n) const { return v_+n; }

    bool operator==(counter x) const { return v_==x.v_; }
    bool operator!=(counter x) const { return v_!=x.v_; }
    bool operator<=(counter x) const { return v_<=x.v_; }
    bool operator>=(counter x) const { return v_>=x.v_; }
    bool operator<(counter x) const { return v_<x.v_; }
    bool operator>(counter x) const { return v_>x.v_; }

    counter& operator=(const counter&) = default;
    counter& operator=(counter&&) = default;

private:
    V v_;
};

} // namespace util
} // namespace arb
