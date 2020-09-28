#pragma once

/*
 * An iterator adaptor that lazily skips items not matching a predicate.
 */

#include <iterator>
#include <memory>
#include <type_traits>

#include <arbor/assert.hpp>
#include <arbor/util/uninitialized.hpp>

#include "util/iterutil.hpp"
#include "util/meta.hpp"
#include "util/range.hpp"

namespace arb {
namespace util {

namespace impl {
    template <typename I, bool = std::is_pointer<I>::value>
    struct arrow {
        using type = decltype(std::declval<I>().operator->());
        static type eval(I& x) { return x.operator->(); }
    };

    template <typename I>
    struct arrow<I, true> {
        using type = I;
        static type eval(I& x) { return x; }
    };
}

/*
 * Iterate through a sequence such that dereference only
 * gives items from the range that satisfy a given predicate.
 *
 * Type parameters:
 *     I      Iterator type
 *     S      Sentinel type compatible with I
 *     F      Functional object
 *
 * The underlying sequence is described by an iterator of type
 * I and a sentinel of type S. The predicate has type F.
 */

template <typename I, typename S, typename F>
class filter_iterator {
    mutable I inner_;
    S end_;
    mutable bool ok_;

    // F may be a lambda type, and thus non-copy assignable. The
    // use of `uninitalized` allows us to work around this limitation;
    // f_ will always be in an initalized state post-construction.
    mutable uninitialized<F> f_;

    void advance() const {
        if (ok_) return;

        for (;;) {
            ok_ = inner_==end_ || f_.ref()(*inner_);
            if (ok_) break;
            ++inner_;
        }
    }

public:
    using value_type = typename std::iterator_traits<I>::value_type;
    using difference_type = typename std::iterator_traits<I>::difference_type;
    using iterator_category =
        std::conditional_t<
            is_forward_iterator<I>::value,
            std::forward_iterator_tag,
            std::input_iterator_tag
        >;

    using pointer = typename std::iterator_traits<I>::pointer;
    using reference = typename std::iterator_traits<I>::reference;

    filter_iterator(): ok_{inner_==end_} {}

    template <typename J, typename K, typename G>
    filter_iterator(J&& iter, K&& end, G&& f):
        inner_(std::forward<J>(iter)),
        end_(std::forward<K>(end)),
        ok_{inner_==end_}
    {
        f_.construct(std::forward<G>(f));
    }

    filter_iterator(const filter_iterator& other):
        inner_(other.inner_),
        end_(other.end_),
        ok_{other.ok_}
    {
        f_.construct(other.f_.cref());
    }

    filter_iterator(filter_iterator&& other):
        inner_(std::move(other.inner_)),
        end_(std::move(other.end_)),
        ok_{other.ok_}
    {
        f_.construct(std::move(other.f_.ref()));
    }

    filter_iterator& operator=(filter_iterator&& other) {
        if (this!=&other) {
            inner_ = std::move(other.inner_);
            end_ = std::move(other.end_);
            ok_ = other.ok_;
            f_.destruct();
            f_.construct(std::move(other.f_.ref()));
        }
        return *this;
    }

    filter_iterator& operator=(const filter_iterator& other) {
        if (this!=&other) {
            inner_ = other.inner_;
            end_ = other.end_;
            ok_ = other.ok_;
            f_.destruct();
            f_.construct(other.f_.cref());
        }
        return *this;
    }

    // forward and input iterator requirements

    decltype(auto) operator*() const {
        advance();
        return *inner_;
    }

    typename impl::arrow<I>::type
    operator->() const {
        advance();
        return impl::arrow<I>::eval(inner_);
    }

    filter_iterator& operator++() {
        advance();
        ok_ = false;
        ++inner_;
        return *this;
    }

    filter_iterator operator++(int) {
        auto c(*this);
        ++*this;
        advance();
        return c;
    }

    bool operator==(const filter_iterator& other) const {
        advance();
        other.advance();
        return inner_==other.inner_;
    }

    bool operator!=(const filter_iterator& other) const {
        return !(*this==other);
    }

    // expose inner iterator for testing against a sentinel
    template <typename Sentinel>
    bool operator==(const Sentinel& s) const {
        advance();
        return inner_==s;
    }

    template <typename Sentinel>
    bool operator!=(const Sentinel& s) const { return !(inner_==s); }

    // public access to inner iterator
    const I& get() const {
        advance();
        return inner_;
    }
};

template <typename I, typename S, typename F>
filter_iterator<I, S, std::decay_t<F>> make_filter_iterator(const I& i, const S& end, const F& f) {
    return filter_iterator<I, S, std::decay_t<F>>(i, end, f);
}

template <typename Seq, typename F>
auto filter(Seq&& s, const F& f) {
    using std::begin;
    using std::end;

    auto b = begin(s);
    auto e = end(s);

    if constexpr (is_regular_sequence_v<Seq&&>) {
        return make_range(make_filter_iterator(b, e, f), make_filter_iterator(e, e, f));
    }
    else {
        return make_range(make_filter_iterator(b, e, f), e);
    }
}

} // namespace util
} // namespace arb
