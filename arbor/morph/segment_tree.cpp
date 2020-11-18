#include <iostream>
#include <stdexcept>
#include <unordered_map>
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
    seg_children_.reserve(n);
}

msize_t segment_tree::append(msize_t p, const mpoint& prox, const mpoint& dist, int tag) {
    if (p>=size() && p!=mnpos) {
        throw invalid_segment_parent(p, size());
    }

    auto id = size();
    segments_.push_back(msegment{id, prox, dist, tag});
    parents_.push_back(p);

    // Set the point properties for the new point, and update those of the parent.
    seg_children_.push_back({});
    if (p!=mnpos) {
        seg_children_[p].increment();
    }

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

bool segment_tree::is_fork(msize_t i) const {
    if (i>=size()) throw no_such_segment(i);
    return seg_children_[i].is_fork();
}
bool segment_tree::is_terminal(msize_t i) const {
    if (i>=size()) throw no_such_segment(i);
    return seg_children_[i].is_terminal();
}
bool segment_tree::is_root(msize_t i) const {
    if (i>=size()) throw no_such_segment(i);
    return parents_[i]==mnpos;
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

} // namespace arb

