#pragma once

/*
 * An iterator adaptor that presents the values from an underlying
 * iterator after applying a provided functor.
 */

#include <iterator>
#include <memory>
#include <type_traits>

#include <util/iterutil.hpp>
#include <util/meta.hpp>
#include <util/range.hpp>

namespace nest {
namespace mc {
namespace util {

template <typename I, typename F>
class transform_iterator: public iterator_adaptor<transform_iterator<I, F>, I> {
    using base = iterator_adaptor<transform_iterator<I, F>, I>;
    friend class iterator_adaptor<transform_iterator<I, F>, I>;

    I inner_;
    F f_;

    // provides access to inner iterator for adaptor.
    const I& inner() const { return inner_; }
    I& inner() { return inner_; }

    using inner_value_type = util::decay_t<decltype(*inner_)>;

public:
    using typename base::difference_type;
    using value_type = typename std::result_of<F (inner_value_type)>::type;
    using pointer = const value_type*;
    using reference = const value_type&;

    template <typename J, typename G>
    transform_iterator(J&& c, G&& g): inner_(std::forward<J>(c)), f_(std::forward<G>(g)) {}

    transform_iterator(const transform_iterator&) = default;
    transform_iterator(transform_iterator&&) = default;
    transform_iterator& operator=(const transform_iterator&) = default;
    transform_iterator& operator=(transform_iterator&&) = default;

    // forward and input iterator requirements

    value_type operator*() const {
        return f_(*inner_);
    }

    util::pointer_proxy<value_type> operator->() const {
        return **this;
    }

    value_type operator[](difference_type n) const {
        return *(*this+n);
    }

    // public access to inner iterator
    const I& get() const { return inner_; }

    bool operator==(const transform_iterator& x) const { return inner_==x.inner_; }
    bool operator!=(const transform_iterator& x) const { return inner_!=x.inner_; }

    // expose inner iterator for testing against a sentinel
    template <typename Sentinel>
    bool operator==(const Sentinel& s) const { return inner_==s; }

    template <typename Sentinel>
    bool operator!=(const Sentinel& s) const { return !(inner_==s); }
};

template <typename I, typename F>
transform_iterator<I, util::decay_t<F>> make_transform_iterator(const I& i, const F& f) {
    return transform_iterator<I, util::decay_t<F>>(i, f);
}

template <
    typename Seq,
    typename F,
    typename seq_citer = typename sequence_traits<Seq>::const_iterator,
    typename seq_csent = typename sequence_traits<Seq>::const_sentinel,
    typename = enable_if_t<std::is_same<seq_citer, seq_csent>::value>
>
range<transform_iterator<seq_citer, util::decay_t<F>>>
transform_view(const Seq& s, const F& f) {
    return {make_transform_iterator(cbegin(s), f), make_transform_iterator(cend(s), f)};
}


template <
    typename Seq,
    typename F,
    typename seq_citer = typename sequence_traits<Seq>::const_iterator,
    typename seq_csent = typename sequence_traits<Seq>::const_sentinel,
    typename = enable_if_t<!std::is_same<seq_citer, seq_csent>::value>
>
range<transform_iterator<seq_citer, util::decay_t<F>>, seq_csent>
transform_view(const Seq& s, const F& f) {
    return {make_transform_iterator(cbegin(s), f), cend(s)};
}


} // namespace util
} // namespace mc
} // namespace nest
