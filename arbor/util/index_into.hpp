#pragma once

// Iterator and range that represents the indices within a super-sequence
// of elements in a sub-sequence.
//
// It is a prerequisite that the elements of the sub-sequence do indeed
// exist within the super-sequence with the same order.
//
// Example:
//
// Given sequence S = { 1, 3, 5, 5, 2 }
// and T = { 0, 5, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 2, 8 },
// then index_into(S, T) would present the indices
// { 3, 4, 8, 8, 13 }.

#include <iterator>
#include <type_traits>

#include <arbor/assert.hpp>

#include "util/meta.hpp"
#include "util/range.hpp"

namespace arb {
namespace util {

template <typename Sub, typename Sup, typename SupEnd>
struct index_into_iterator {
    using value_type = typename std::iterator_traits<Sup>::difference_type;
    using difference_type = value_type;
    using pointer = const value_type*;
    using reference = const value_type&;
    using iterator_category =
        std::conditional_t<
            std::is_same_v<Sup, SupEnd>
                && is_bidirectional_iterator_v<Sup>
                && is_bidirectional_iterator_v<Sub>,
            std::bidirectional_iterator_tag,
            std::forward_iterator_tag
        >;

    index_into_iterator(const Sub& sub, const Sub& sub_end, const Sup& sup, const SupEnd& sup_end):
        sub(sub), sub_end(sub_end), sup(sup), sup_end(sup_end), idx(0)
    {
        align_fwd();
    }

    value_type operator*() const {
        return idx;
    }

    index_into_iterator& operator++() {
        arb_assert(sup!=sup_end);

        ++sub;
        align_fwd();
        return *this;
    }

    index_into_iterator operator++(int) {
        auto keep = *this;
        ++(*this);
        return keep;
    }

    index_into_iterator& operator--() {
        if (sub==sub_end) {
            // decrementing one-past-the-end iterator
            idx = std::distance(sup, sup_end)-1;
            sup = std::prev(sup_end);
        }

        --sub;
        align_rev();
        return *this;
    }

    index_into_iterator operator--(int) {
        auto keep = *this;
        --(*this);
        return keep;
    }

    template <typename A, typename C, typename D>
    friend struct index_into_iterator;

    template <typename OSub>
    bool operator==(const index_into_iterator<OSub, Sup, SupEnd>& other) const {
        return sub==other.sub;
    }

    template <typename OSub>
    bool operator!=(const index_into_iterator<OSub, Sup, SupEnd>& other) const {
        return !(*this==other);
    }

private:
    Sub sub;
    Sub sub_end;
    Sup sup;
    SupEnd sup_end;
    difference_type idx;

    void align_fwd() {
        if (sub!=sub_end) {
            while (sup!=sup_end && !(*sub==*sup)) {
                ++idx;
                ++sup;
            }
        }
    }

    void align_rev() {
        while (idx>0 && !(*sub==*sup)) {
            --idx;
            --sup;
        }

        arb_assert(*sub==*sup);
    }
};

template <typename Sub, typename Super>
auto index_into(const Sub& sub, const Super& sup) {
    using std::begin;
    using std::end;

    auto canon = canonical_view(sub);
    using iterator = index_into_iterator<decltype(canon.begin()), decltype(begin(sup)), decltype(end(sup))>;

    return make_range(
        iterator(canon.begin(), canon.end(), begin(sup), end(sup)),
        iterator(canon.end(), canon.end(), begin(sup), end(sup))
    );
}

} // namespace util
} // namespace arb
