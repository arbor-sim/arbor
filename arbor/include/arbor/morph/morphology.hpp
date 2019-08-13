#pragma once

#include <ostream>
#include <vector>

#include <arbor/util/counter.hpp>
#include <arbor/util/lexcmp_def.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/sample_tree.hpp>

namespace arb {

template <typename T, typename I>
struct indexer: util::counter<I> {
    using base = util::counter<I>;
    using difference_type = typename base::difference_type;

    // Hold the functor that will transform the underlying counter to the index.
    std::function<T(I)> map;

    indexer(std::function<T(I)> f, I i): base(i), map(std::move(f)) {};
    indexer(std::function<T(I)> f): indexer(std::move(f), I{0}) {};
    indexer(): indexer([](const I& i){return i;}, I{0}) {};

    // Dereferencing the iterator requrires transforming the underlying counter.
    T operator*() const {return map(index());}
    T operator[](I i) const {return map(i);}

    // Helper for accessing the value of the underlying counter.
    I index() const {return base::operator*();}

    // Members of counter that return or operate on the counter must be
    // to perform the equivalent operation for indexer.

    indexer operator+(difference_type n) {
        indexer x(*this);
        x+=n;
        return x;
    }

    friend indexer operator+(difference_type n, indexer x) {
        x += n;
        return x;
    }

    indexer operator-(difference_type n) {
        indexer x(*this);
        x-=n;
        return x;
    }

    difference_type operator-(indexer x) const {
        return index()-x.index();
    }
};

class morphology {
    sample_tree sample_tree_;

    // Branch state.
    std::vector<mbranch> branches_;
    std::vector<size_t> branch_parents_;
    std::vector<std::vector<size_t>> branch_children_;

    // Meta data about sample point properties.
    std::vector<size_t> fork_points_;
    std::vector<size_t> terminal_points_;
    std::vector<point_prop> point_props_;

    // Indicates whether the soma is a sphere.
    bool spherical_root_;

    // Types used to provide range-based access to indexes and samples in branches.
    using index_counter = indexer<size_t, size_t>;
    using sample_counter = indexer<msample, size_t>;
    using index_range = std::pair<index_counter, index_counter>;
    using sample_range = std::pair<sample_counter, sample_counter>;

public:
    morphology(sample_tree m);

    // Whether the root of the morphology is spherical.
    bool spherical_root() const;

    // The number of branches in the morphology.
    size_t num_branches() const;

    // List the ids of fork points in the morphology.
    const std::vector<size_t>& fork_points() const;

    // List the ids of terminal points in the morphology.
    const std::vector<size_t>& terminal_points() const;

    // The parent branch of branch b.
    size_t branch_parent(size_t b) const;

    // The child branches of branch b.
    const std::vector<size_t>& branch_children(size_t b) const;

    // Range of indexes into the sample points in branch b.
    index_range branch_sample_span(size_t b) const;

    // Range of the samples in branch b.
    sample_range branch_sample_view(size_t b) const;

    friend std::ostream& operator<<(std::ostream&, const morphology&);
};

} // namespace arb
