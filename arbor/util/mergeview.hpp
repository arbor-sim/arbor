#pragma once

#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "util/iterutil.hpp"
#include "util/meta.hpp"
#include "util/range.hpp"

namespace arb {
namespace util {

template <typename A, typename B, typename = void>
struct has_common_type: std::false_type {};

template <typename A, typename B>
struct has_common_type<A, B, std::void_t<std::common_type_t<A, B>>>: std::true_type {};

template <typename A, typename B, typename X, typename = void>
struct common_type_or_else { using type = X; };

template <typename A, typename B, typename X>
struct common_type_or_else<A, B, X, std::void_t<std::common_type_t<A, B>>> {
    using type = std::common_type_t<A, B>;
};

template <typename A, typename B, typename X>
using common_type_or_else_t = typename common_type_or_else<A, B, X>::type;

template <typename LeftI, typename LeftS, typename RightI, typename RightS>
class merge_iterator {
    mutable LeftI left_;
    mutable RightI right_;

    LeftS left_sentinel_;
    RightS right_sentinel_;

    mutable enum which {
        undecided, left_next, right_next, done
    } next_ = undecided;

    void resolve_next() const {
        if (next_!=undecided) return;

        if (left_==left_sentinel_) {
            next_ = right_==right_sentinel_? done: right_next;
        }
        else if (right_==right_sentinel_) {
            next_ = left_next;
        }
        else {
            next_ = *left_<*right_? left_next: right_next;
        }
    }

public:
    using value_type = std::common_type_t<
        typename std::iterator_traits<LeftI>::value_type,
        typename std::iterator_traits<RightI>::value_type>;

    using pointer = common_type_or_else_t<LeftI, RightI, pointer_proxy<value_type>>;

    using reference = std::conditional_t<
        has_common_type<LeftI, RightI>::value,
        typename std::iterator_traits<common_type_or_else_t<LeftI, RightI, char*>>::reference,
        value_type>;

    using difference_type = typename std::iterator_traits<LeftI>::difference_type;
    using iterator_category = std::forward_iterator_tag;

    merge_iterator(): next_(done) {}

    template <typename LSeq, typename RSeq>
    merge_iterator(LSeq&& lseq, RSeq&& rseq) {
        using std::begin;
        using std::end;

        left_ = begin(lseq);
        left_sentinel_ = end(lseq);
        right_ = begin(rseq);
        right_sentinel_ = end(rseq);
    }

    bool operator==(const merge_iterator& x) const {
        resolve_next();
        x.resolve_next();

        if (next_==done && x.next_==done) return true;
        return next_==x.next_ && left_==x.left_ && right_==x.right_;
    }

    bool operator!=(const merge_iterator& x) const { return !(*this==x); }

    reference operator*() const {
        resolve_next();

        switch (next_) {
        case left_next: return *left_;
        case right_next: return *right_;
        default:
            throw std::range_error("derefence past end of sequence");
        }
    }

    template <typename L = LeftI, typename R = RightI,
              typename = std::enable_if_t<has_common_type<L, R>::value>>
    std::common_type_t<L, R> operator->() const {
        resolve_next();
        return next_==left_next? left_: right_;
    }

    template <typename L = LeftI, typename R = RightI,
              typename = std::enable_if_t<!has_common_type<L, R>::value>>
    pointer_proxy<value_type> operator->() const {
        return pointer_proxy<value_type>(*this);
    }

    merge_iterator& operator++() {
        resolve_next();

        switch (next_) {
        case undecided:
            throw std::logic_error("internal error: unexpected undecided");
        case done: return *this;
        case left_next:
            ++left_;
            break;
        case right_next:
            ++right_;
            break;
        }
        next_ = undecided;
        return *this;
    }

    merge_iterator operator++(int) {
        auto me = *this;
        ++*this;
        return me;
    }

    merge_iterator& operator=(const merge_iterator&) = default;
};

template <typename LSeq, typename RSeq>
auto merge_view(LSeq&& left, RSeq&& right) {
    using std::begin;
    using std::end;
    using LIter = decltype(begin(left));
    using LSentinel = decltype(end(left));
    using RIter = decltype(begin(right));
    using RSentinel = decltype(end(right));
    using MIter = merge_iterator<LIter, LSentinel, RIter, RSentinel>;

    return make_range(MIter(left, right), MIter());
}


} // namespace util
} // namespace arb

