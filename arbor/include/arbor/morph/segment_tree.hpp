#pragma once

#include <cassert>
#include <functional>
#include <vector>
#include <string>

#include <arbor/export.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

/// Morphology composed of segments.
class ARB_ARBOR_API segment_tree {
    struct child_prop {
        int count;
        bool is_fork() const { return count>1; }
        bool is_terminal() const { return count==0; }
        int increment() { return ++count;}
    };

    std::vector<msegment> segments_;
    std::vector<msize_t> parents_;
    std::vector<child_prop> seg_children_;

public:
    segment_tree() = default;

    // Reserve space for n segments.
    void reserve(msize_t n);

    // The append functions return a handle to the last segment appended by the call.

    // Append a single segment.
    msize_t append(msize_t p, const mpoint& prox, const mpoint& dist, int tag);
    msize_t append(msize_t p, const mpoint& dist, int tag);

    // The number of segments in the tree.
    msize_t size() const;
    bool empty() const;

    // The segments in the tree.
    const std::vector<msegment>& segments() const;

    // The parent index of the segments.
    const std::vector<msize_t>& parents() const;

    // Interfaces for querying the properties of the segments by index.

    bool is_fork(msize_t i) const;
    bool is_terminal(msize_t i) const;
    bool is_root(msize_t i) const;

    friend std::ostream& operator<<(std::ostream&, const segment_tree&);

    // compare two trees for _identity_, not _equivalence_
    friend bool operator==(const segment_tree& l, const segment_tree& r) {
        return (l.size() == r.size()) && (l.parents() == r.parents()) && (l.segments() == r.segments());
    }
};

// Split a segment_tree T into two subtrees <L, R> such that R is the subtree
// of T that starts at the given id and L is T without R.
std::pair<segment_tree, segment_tree>
split_at(const segment_tree&, msize_t);

// Join two subtrees L and R at a given id in L, such that `join_at` is inverse
// to `split_at` for a particular choice of id.
segment_tree
join_at(const segment_tree&, msize_t, const segment_tree&);

} // namesapce arb
