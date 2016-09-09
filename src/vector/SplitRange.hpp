#pragma once

#include <iterator>
#include <ostream>

#include <cassert>

#include "Range.hpp"

namespace memory {

class SplitRange {
  public:
    using size_type       = Range::size_type;
    using difference_type = Range::difference_type;

    // split range into n chunks
    SplitRange(Range const& rng, size_type n) : range_(rng) {
        // it makes no sense to break a range into 0 chunks
        assert(n>0);

        // add one to step_ if n does not evenly subdivide the target range
        step_ = rng.size()/n + (rng.size()%n ? 1 : 0);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Iterator to generate a sequence of ranges that split the range into
    // n disjoint sets
    //
    // Derived from input_iterator because it only makes sense to use as a read
    // only iterator: the iterator does not refer to any external memory,
    // instead returns to state that can only change when iterator is
    // incremented via ++ operator
    ///////////////////////////////////////////////////////////////////////////
    class iterator
      : public std::iterator<std::random_access_iterator_tag, Range>
    {
      public:
          iterator(size_type first, size_type end, size_type step)
              : range_(first, first+step),
                step_(step),
                begin_(first),
                end_(end)
          {
              assert(first<=end);

              if(range_.right()>end) {
                  range_.set(first, end);
              }
          }

          Range const& operator*() const {
              return range_;
          }

          Range const* operator->() const {
              return &range_;
          }

          iterator operator++(int) {
              iterator previous(*this);
              ++(*this);
              return previous;
          }

          iterator operator--(int) {
              iterator next(*this);
              --(*this);
              return next;
          }

          const iterator* operator++() {
              size_type first = range_.left()+step_;
              if(first>end_)
                  first=end_;

              size_type last = range_.right()+step_;
              if(last>end_)
                  last=end_;

              // update range
              range_.set(first, last);

              return this;
          }

          const iterator* operator--() {
              size_type first = range_.left()-step_;
              if(first<begin_)
                  first=begin_;

              size_type last = first+step_;
              if(last>end_)
                  last=end_;

              // update range
              range_.set(first, last);

              return this;
          }

          const iterator* operator+=(int n) {
              size_type first = range_.left()+step_*n;
              if(first>end_)
                  first=end_;

              size_type last = first+step_;
              if(last>end_)
                  last=end_;

              // update range
              range_.set(first, last);

              return this;
          }

          iterator operator+(int n) {
              iterator i(*this);
              i+=n;
              return i;
          }

          const iterator* operator-=(int n) {
              size_type first = range_.left()+step_*n;
              if(first<begin_)
                  first=begin_;

              size_type last = first+step_;
              if(last>end_)
                  last=end_;

              // update range
              range_.set(first, last);

              return this;
          }

          iterator operator-(int n) {
              iterator i(*this);
              i-=n;
              return i;
          }

          bool operator == (const iterator& other) const {
              return range_ == other.range_;
          }

          bool operator != (const iterator& other) const {
              return range_ != other.range_;
          }

      private:
        Range range_;
        size_type begin_;  // first value for begin_
        size_type end_;    // final value for end
        size_type step_;   // step by which range limits get increased
    };
    ///////////////////////////////////////

    iterator begin() const {
        return iterator(range_.left(), range_.right(), step_);
    }

    iterator end() const {
        return iterator(range_.right(), range_.right(), step_);
    }

    Range operator [] (int i) const {
        return *(begin()+i);
    }

    size_type step_size() const {
        return step_;
    }

    Range range() const {
        return range_;
    }

  private:
    size_type step_;
    Range range_;
};

// overload output operator for split range
static std::ostream& operator << (std::ostream& os, const SplitRange& split) {
    os << "(" << split.range() << " by " << split.step_size() << ")";
    return os;
}

}
