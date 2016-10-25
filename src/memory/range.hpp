#pragma once

#include <ostream>

#include <cassert>

#include "range_limits.hpp"

#ifdef USING_TBB
    #include <tbb/tbb.h>
#endif

namespace nest {
namespace mc {
namespace memory {

class Range {
public:
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    Range()
    : left_(0), right_(0)
    {}

    explicit Range(size_type n)
    : left_(0), right_(n)
    {}

    Range(size_type b, size_type e)
    : left_(b), right_(e)
    {}

    //
    // make Range compatible with TBB Range concept
    //
#ifdef USING_TBB
    bool empty() const {
        return right_==left_;
    }

    bool is_divisible() const {
        return size()>1;
    }

    Range(Range& other, tbb::split) {
        auto m = (other.left() + other.right())/2;
        left_ = m;
        right_ = other.right();
        other.set(other.left(), m);
    }

    Range(Range& other, tbb::proportional_split p) {
        auto m = ((other.left()+other.right())*p.right())/(p.left()+p.right());
        if(m == 0) {
            m = 1;
        }
        else if(m == other.right()) {
            m = other.right() - 1;
        }

        left_ = m;
        right_ = other.right();
        other.set(other.left(), m);
    }

    static constexpr bool is_splittable_in_proportion = true;
#endif // USING_TBB

    Range(Range const& other) = default;

    size_type size() const {
        return right_ - left_;
    }

    size_type left() const {
        return left_;
    }

    size_type right() const {
        return right_;
    }

    void set(size_type b, size_type e) {
        left_ = b;
        right_ = e;
    }

    Range& operator +=(size_type n) {
        left_ += n;
        right_ += n;

        return (*this);
    }

    Range& operator -=(size_type n) {
        left_ -= n;
        right_   -= n;

        return (*this);
    }

    bool operator == (const Range& other) const {
        return (left_ == other.left_) && (right_ == other.right_);
    }

    bool operator != (const Range& other) const {
        return (left_ != other.left_) || (right_ != other.right_);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Iterator to generate a sequence of integral values
    //
    // Derived from input_iterator because it only makes sense to use as a read
    // only iterator: the iterator does not refer to any external memory,
    // instead returns to state that can only change when iterator is
    // incremented via ++ operator
    ///////////////////////////////////////////////////////////////////////////
    class iterator
    : public std::iterator<std::input_iterator_tag, size_type>
    {
    public:
        iterator(size_type first)
        : index_(first)
        {}

        size_type const& operator*() const {
            return index_;
        }

        size_type const* operator->() const {
            return &index_;
        }

        iterator operator++(int) {
            iterator previous(*this);
            ++(*this);
            return previous;
        }

        const iterator* operator++() {
            ++index_;
            return this;
        }

        bool operator == (const iterator& other) const {
            return index_ == other.index_;
        }

        bool operator != (const iterator& other) const {
            return index_ != other.index_;
        }

    private:
        size_type index_;
    };
    ///////////////////////////////////////

    iterator begin() const {
        return iterator(left_);
    }

    iterator end() const {
        return iterator(right_);
    }

    // operators to get sub-ranges
    Range operator () (size_type left, size_type right) const {
        #ifndef NDEBUG
        assert(left_+left  <= right_);
        assert(left_+right <= right_);
        #endif
        return Range(left_+left, left_+right);
    }

    Range operator () (size_type left, memory::end_type) const {
        #ifndef NDEBUG
        assert(left_+left  <= right_);
        #endif
        return Range(left_+left, right_);
    }

private:
    size_type left_;
    size_type right_;
};

static inline std::ostream& operator << (std::ostream& os, const Range& rng) {
    os << "[" << rng.left() << ":" << rng.right() << "]";
    return os;
}

} // namespace memory
} // namespace mc
} // namespace nest
