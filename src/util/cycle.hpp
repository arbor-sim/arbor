#pragma once

#include <utility>
#include <util/iterutil.hpp>
#include <util/range.hpp>

namespace nest {
namespace mc {
namespace util {

template <typename I, typename S = I>
class cyclic_iterator : public iterator_adaptor<cyclic_iterator<I,S>, I> {
    using base = iterator_adaptor<cyclic_iterator<I,S>, I>;
    friend class iterator_adaptor<cyclic_iterator<I,S>, I>;

    I begin_;
    I inner_;
    S end_;
    typename base::difference_type size_;  // wrap distance

    const I& inner() const {
        return inner_;
    }

    I& inner() {
        return inner_;
    }

public:
    using value_type = typename base::value_type;
    using difference_type = typename base::difference_type;

    cyclic_iterator() = default;

    template <typename Iter, typename Sentinel>
    cyclic_iterator(Iter&& iter, Sentinel&& sentinel)
        : begin_(std::forward<Iter>(iter)),
          inner_(std::forward<Iter>(iter)),
          end_(std::forward<Sentinel>(sentinel)),
          size_(util::distance(iter, sentinel))
    { }

    cyclic_iterator(const cyclic_iterator& other)
        : begin_(other.begin_),
          inner_(other.inner_),
          end_(other.end_),
          size_(other.size_) { }

    cyclic_iterator(cyclic_iterator&& other)
        : begin_(std::move(other.begin_)),
          inner_(std::move(other.inner_)),
          end_(std::move(other.end_)),
          size_(other.size_) { }


    cyclic_iterator& operator=(const cyclic_iterator& other) {
        if (this != &other) {
            inner_ = other.inner_;
            begin_ = other.begin_;
            end_ = other.end_;
            size_ = other.size_;
        }

        return *this;
    }

    cyclic_iterator& operator=(cyclic_iterator&& other) {
        if (this != &other) {
            inner_ = std::move(other.inner_);
            begin_ = std::move(other.begin_);
            end_ = std::move(other.end_);
            size_ = other.size_;
        }

        return *this;
    }

    // forward and input iterator requirements
    value_type operator*() const {
        return *inner_;
    }

    value_type operator[](difference_type n) const {
        return *(*this + n);
    }

    cyclic_iterator& operator++() {
        if (++inner_ == end_) {
            // wrap around
            inner_ = begin_;
        }

        return *this;
    }

    cyclic_iterator operator++(int) {
        cyclic_iterator iter(*this);
        ++(*this);
        return iter;
    }

    cyclic_iterator& operator--() {
        if (inner_ == begin_) {
            // wrap around
            inner_ = std::next(begin_, size_-1);
        }
        else {
            --inner_;
        }

        return *this;
    }

    cyclic_iterator operator--(int) {
        cyclic_iterator iter(*this);
        --(*this);
        return iter;
    }

    cyclic_iterator& operator+=(difference_type n) {
        // calculate distance from begin
        auto pos = util::distance(begin_, inner_) + n;
        if (pos < 0) {
            auto mod = -pos % size_;
            pos = mod ? size_ - mod : 0;
        }
        else {
            pos = pos % size_;
        }

        inner_ = std::next(begin_, pos);
        return *this;
    }

    cyclic_iterator& operator-=(difference_type n) {
        return this->operator+=(-n);
    }

    bool operator==(const cyclic_iterator& other) const {
        return inner_ == other.inner_;
    }

    bool operator!=(const cyclic_iterator& other) const {
        return !(*this == other);
    }

    // expose inner iterator for testing against a sentinel
    template <typename Sentinel>
    bool operator==(const Sentinel& s) const { return inner_ == s; }

    template <typename Sentinel>
    bool operator!=(const Sentinel& s) const { return !(inner_ == s); }
};

template <typename I, typename S>
cyclic_iterator<I, S> make_cyclic_iterator(const I& iter, const S& sentinel) {
    return cyclic_iterator<I, S>(iter, sentinel);
}


template <
    typename Seq,
    typename SeqIter = typename sequence_traits<Seq>::const_iterator,
    typename SeqSentinel = typename sequence_traits<Seq>::const_sentinel,
    typename = enable_if_t<std::is_same<SeqIter, SeqSentinel>::value>
>
range<cyclic_iterator<SeqIter, SeqSentinel> > cyclic_view(const Seq& s) {
    return { make_cyclic_iterator(cbegin(s), cend(s)),
             make_cyclic_iterator(cend(s), cend(s)) };
}

template <
    typename Seq,
    typename SeqIter = typename sequence_traits<Seq>::const_iterator,
    typename SeqSentinel = typename sequence_traits<Seq>::const_sentinel,
    typename = enable_if_t<!std::is_same<SeqIter, SeqSentinel>::value>
>
range<cyclic_iterator<SeqIter, SeqSentinel>, SeqSentinel>
cyclic_view(const Seq& s) {
    // iterating over a cyclic view is endless; the following two iterators can
    // never be equal
    return { make_cyclic_iterator(cbegin(s), cend(s)), cend(s) };
}

} // namespace util
} // namespace mc
} // namespace nest
