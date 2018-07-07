#pragma once

/*
 * Present a monotonically increasing sequence of values given by an iterator
 * as a sequence of pairs representing half-open intervals.
 *
 * Implementation is a thin wrapper over underlying iterator.
 */

#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include <util/iterutil.hpp>
#include <util/meta.hpp>

namespace arb {
namespace util {

template <typename I>
class partition_iterator: public iterator_adaptor<partition_iterator<I>, I> {
    // TODO : dirty workaround to make inner_ public by making everything public
public:
    using base = iterator_adaptor<partition_iterator<I>, I>;
    friend class iterator_adaptor<partition_iterator<I>, I>;

    I inner_;

    // provides access to inner iterator for adaptor.
    const I& inner() const { return inner_; }
    I& inner() { return inner_; }

    using inner_value_type = std::decay_t<decltype(*inner_)>;

    using typename base::difference_type;
    using value_type = std::pair<inner_value_type, inner_value_type>;
    using pointer = const value_type*;
    using reference = const value_type&;

    partition_iterator() = default;

    template <
        typename J,
        typename = std::enable_if_t<!std::is_same<std::decay_t<J>, partition_iterator>::value>
    >
    explicit partition_iterator(J&& c): inner_{std::forward<J>(c)} {}

    // forward and input iterator requirements

    value_type operator*() const {
        return {*inner_, *std::next(inner_)};
    }

    util::pointer_proxy<value_type> operator->() const {
        return **this;
    }

    value_type operator[](difference_type n) const {
        return *(*this+n);
    }

    // public access to inner iterator
    const I& get() const { return inner_; }
};

} // namespace util
} // namespace arb
