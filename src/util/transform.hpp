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

/* Note, this is actually only an input iterator if F is non-assignable, such
 * as when it is a lambda! */

template <typename I, typename F>
class transform_iterator: public iterator_adaptor<transform_iterator<I, F>, I> {
    using base = iterator_adaptor<transform_iterator<I, F>, I>;
    friend class iterator_adaptor<transform_iterator<I, F>, I>;

    I inner_;

    // F may be a lambda type, and thus non-copy assignable. The
    // use of `uninitalized` allows us to work around this limitation;
    uninitialized<F> f_;

    // provides access to inner iterator for adaptor.
    const I& inner() const { return inner_; }
    I& inner() { return inner_; }

    using inner_value_type = decltype(*inner_);
    using raw_value_type = typename std::result_of<F (inner_value_type)>::type;

    static constexpr bool present_lvalue = std::is_reference<raw_value_type>::value;


public:
    using typename base::difference_type;
    using value_type = util::decay_t<raw_value_type>;
    using pointer = typename std::conditional<present_lvalue, value_type*, const value_type*>::type;
    using reference = typename std::conditional<present_lvalue, raw_value_type, const value_type&>::type;

    transform_iterator() = default;

    template <typename J, typename G>
    transform_iterator(J&& c, G&& g): inner_(std::forward<J>(c)) {
        f_.construct(std::forward<G>(g));
    }

    transform_iterator(const transform_iterator& other): inner_(other.inner_) {
        f_.construct(other.f_.cref());
    }

    transform_iterator(transform_iterator&& other): inner_(std::move(other.inner_)) {
        f_.construct(std::move(other.f_.ref()));
    }

    transform_iterator& operator=(transform_iterator&& other) {
        if (this!=&other) {
            inner_ = std::move(other.inner_);
            f_.construct(std::move(other.f_.ref()));
        }
        return *this;
    }

    transform_iterator& operator=(const transform_iterator& other) {
        if (this!=&other) {
            inner_ = other.inner_;
            f_.destruct();
            f_.construct(other.f_.cref());
        }
        return *this;
    }

    // forward and input iterator requirements

    typename std::conditional<present_lvalue, reference, value_type>::type
    operator*() const {
        return f_.cref()(*inner_);
    }

    typename std::conditional<present_lvalue, pointer, util::pointer_proxy<value_type>>::type
    operator->() const {
        return pointer_impl(std::integral_constant<bool, present_lvalue>{});
    }

    typename std::conditional<present_lvalue, reference, value_type>::type
    operator[](difference_type n) const {
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

private:
    // helper routines for operator->(): need different implementations for
    // lvalue and non-lvalue access.
    util::pointer_proxy<value_type> pointer_impl(std::false_type) const {
        return **this;
    }

    pointer pointer_impl(std::true_type) const {
        return &(**this);
    }
};

template <typename I, typename F>
transform_iterator<I, util::decay_t<F>> make_transform_iterator(const I& i, const F& f) {
    return transform_iterator<I, util::decay_t<F>>(i, f);
}

template <
    typename Seq,
    typename F,
    typename seq_iter = typename sequence_traits<Seq>::iterator,
    typename seq_sent = typename sequence_traits<Seq>::sentinel,
    typename = enable_if_t<std::is_same<seq_iter, seq_sent>::value>
>
range<transform_iterator<seq_iter, util::decay_t<F>>>
transform_view(Seq&& s, const F& f) {
    return {make_transform_iterator(std::begin(s), f), make_transform_iterator(std::end(s), f)};
}

template <
    typename Seq,
    typename F,
    typename seq_iter = typename sequence_traits<Seq>::iterator,
    typename seq_sent = typename sequence_traits<Seq>::sentinel,
    typename = enable_if_t<!std::is_same<seq_iter, seq_sent>::value>
>
range<transform_iterator<seq_iter, util::decay_t<F>>, seq_sent>
transform_view(Seq&& s, const F& f) {
    return {make_transform_iterator(std::begin(s), f), std::end(s)};
}

} // namespace util
} // namespace mc
} // namespace nest
