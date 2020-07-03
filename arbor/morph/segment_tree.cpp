#include <iostream>
#include <stdexcept>
#include <vector>

#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/segment_tree.hpp>

#include "io/sepval.hpp"
#include "util/span.hpp"
#include "util/transform.hpp"

namespace arb {

void segment_tree::reserve(msize_t n) {
    segments_.reserve(n);
    parents_.reserve(n);
    props_.reserve(n);
}

//msize_t segment_tree::append(msize_t p, const mpoint& prox, const mpoint& dist, int tag);
msize_t segment_tree::append(msize_t p, const mpoint& prox, const mpoint& dist, int tag) {
    if (p>=size() && p!=mnpos) {
        throw invalid_segment_parent(p, size());
    }

    auto id = size();
    segments_.push_back(msegment{prox, dist, tag});
    parents_.push_back(p);

    // Set the point properties for the new point, and update those of its parent as needed.
    seg_prop prop = seg_prop_mask_none;
    if (p==mnpos) {
        // If attaching a segment to the root, mark it as root.
        set_seg_root(prop);
    }
    else {
        if (!is_seg_terminal(props_[p])) {
            set_seg_fork(props_[p]);
        }
        // Unset the terminal tag on the parent.
        unset_seg_terminal(props_[p]);
    }
    // Mark the new segment as terminal.
    set_seg_terminal(prop);
    props_.push_back(prop);

    return id;
}

msize_t segment_tree::append(msize_t p, const mpoint& dist, int tag) {
    // If attaching to the root both prox and dist ends must be specified.
    if (p==mnpos) {
        throw invalid_segment_parent(p, size());
    }
    if (p>=size()) {
        throw invalid_segment_parent(p, size());
    }
    return append(p, segments_[p].dist, dist, tag);
}

msize_t segment_tree::size() const {
    return segments_.size();
}

bool segment_tree::empty() const {
    return segments_.empty();
}

const std::vector<msegment>& segment_tree::segments() const {
    return segments_;
}

const std::vector<msize_t>& segment_tree::parents() const {
    return parents_;
}

const std::vector<point_prop>& segment_tree::properties() const {
    return props_;
}

std::ostream& operator<<(std::ostream& o, const segment_tree& m) {
    auto tstr = util::transform_view(m.parents_,
            [](msize_t i) -> std::string {
                return i==mnpos? "npos": std::to_string(i);
            });
    bool one_line = m.size()<2u;
    return o << "(segment_tree (" << (one_line? "": "\n  ") << io::sepval(m.segments_, "\n  ")
             << (one_line? ") (": ")\n  (")
             << io::sepval(tstr, ' ') <<  "))";
}

/*
sample_tree swc_as_sample_tree(const std::vector<swc_record>& swc_records) {
    sample_tree m;
    m.reserve(swc_records.size());

    for (auto i: util::count_along(swc_records)) {
        auto& r = swc_records[i];
        // The parent of soma must be mnpos, while in SWC files is -1
        msize_t p = i==0? mnpos: r.parent_id;
        m.append(p, msample{{r.x, r.y, r.z, r.r}, r.tag});
    }
    return m;
}
*/

} // namespace arb

