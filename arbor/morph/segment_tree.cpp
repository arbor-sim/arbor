#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/segment_tree.hpp>

#include "io/sepval.hpp"
#include "util/span.hpp"
#include "util/transform.hpp"

using arb::util::make_span;

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

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const segment_tree& m) {
    auto tstr = util::transform_view(m.parents_,
            [](msize_t i) -> std::string {
                return i==mnpos? "npos": std::to_string(i);
            });
    bool one_line = m.size()<2u;
    return o << "(segment_tree (" << (one_line? "": "\n  ") << io::sepval(m.segments_, "\n  ")
             << (one_line? ") (": ")\n  (")
             << io::sepval(tstr, ' ') <<  "))";
}


// Utilities.

ARB_ARBOR_API segment_tree prune_tag(const segment_tree& in, int tag) {
    const auto& in_segments = in.segments();
    const auto& in_parents = in.parents();
    segment_tree out;
    std::vector<int> pruned_id_upper_bounds, pruned_id_offsets;

    int num_pruned = 0;
    for (auto i: make_span(0, in_segments.size())) {
        if (in_segments[i].tag == tag) {
            ++num_pruned;
            if (i+1 < in_segments.size() && in_segments[i+1].tag != tag) {
                pruned_id_upper_bounds.push_back(i+1);
                pruned_id_offsets.push_back(num_pruned);
            }
        }
    }

    for (auto i: make_span(in_segments.size())) {
        const auto& seg = in_segments[i];
        auto par = in_parents[i];
        if (seg.tag != tag) {
            if (par != mnpos && in_segments[par].tag == tag) {
                // children of pruned parents must be pruned
                throw unpruned_child(par, seg.id, tag);
            } else {
                if (par != mnpos) {
                    auto ui = upper_bound(pruned_id_upper_bounds.begin(), 
                                          pruned_id_upper_bounds.end(), 
                                          par) - pruned_id_upper_bounds.begin();
                    par -= ui > 0 ? pruned_id_offsets[ui-1] : 0;
                }
                out.append(par, seg.prox, seg.dist, seg.tag);
            }
        }
    }

    return out;
}

ARB_ARBOR_API std::vector<msize_t> prune_tag_roots(const segment_tree& in, int tag) {
    const auto& in_segments = in.segments();
    const auto& in_parents = in.parents();
    std::vector<msize_t> prune_roots;

    for (auto i: make_span(0, in_segments.size())) {
        auto par = in_parents[i];
        if (in_segments[i].tag == tag && (par == mnpos || in_segments[par].tag != tag)) {
            prune_roots.push_back(i);
        }
    }

    return prune_roots;
}


double segment_length(const msegment& seg) {
    auto seg_x = seg.dist.x - seg.prox.x;
    auto seg_y = seg.dist.y - seg.prox.y;
    auto seg_z = seg.dist.z - seg.prox.z;
    return std::sqrt(seg_x*seg_x + seg_y*seg_y + seg_z*seg_z);
}

// TODO: change to median distal radius
ARB_ARBOR_API std::vector<double> median_distal_radii(const segment_tree& in, int tag, double dist) {
    const auto& in_segments = in.segments();
    const auto& in_parents = in.parents();
    auto median_dist = 0.5*dist;

    assert(tag != 1);

    double soma_length = 0.;
    for (auto i: make_span(0, in_segments.size())) {
        if (in_segments[i].tag == 1) {  // soma
            soma_length += segment_length(in_segments[i]);
        }
    }

    dist -= 0.5 * soma_length;
    median_dist -= 0.5 * soma_length;
    if (dist < 0.) {
        return {};
    }

    std::vector<double> distance(in_segments.size(), 0.);
    enum median_rel_pos {before, on, after};
    std::vector<int> median_segment(in_segments.size(), before);
    std::vector<double> out;

    for (auto i: make_span(0, in_segments.size())) {
        auto par = in_parents[i];
        if (in_segments[i].tag == tag) {
            if (par == mnpos) {
                distance[i] = 0.;
            } else if (in_segments[par].tag == tag) {
                distance[i] = distance[par] + segment_length(in_segments[par]);
            }
            auto seg_length = segment_length(in_segments[i]);
            if (distance[i] < median_dist && median_dist < distance[i] + seg_length) {
                median_segment[i] = on;
            } else if (median_dist < distance[i]) {
                median_segment[i] = after;
            }
            if (distance[i] < dist && dist < distance[i] + seg_length) {
                auto median_i = i;
                while (median_segment[median_i] != on) { median_i = in_parents[median_i]; }
                auto t = (median_dist - distance[median_i])/segment_length(in_segments[median_i]);
                out.push_back((1.-t)*in_segments[median_i].prox.radius + t*in_segments[median_i].dist.radius);
            }
        }
    }

    return out;
}


} // namespace arb

