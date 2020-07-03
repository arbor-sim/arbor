#pragma once

#include <cassert>
#include <functional>
#include <vector>
#include <string>

#include <arbor/swcio.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

/// Morphology composed of segments.
class segment_tree {
    std::vector<msegment> segments_;
    std::vector<msize_t> parents_;
    std::vector<seg_prop> props_;

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

    // The properties of the segments.
    const std::vector<seg_prop>& properties() const;

    friend std::ostream& operator<<(std::ostream&, const segment_tree&);
};

/// Build a sample tree from a sequence of swc records.
segment_tree swc_as_segment_tree(const std::vector<swc_record>& swc_records);

} // namesapce arb


