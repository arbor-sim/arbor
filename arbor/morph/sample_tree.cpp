#include <cmath>
#include <vector>

#include <arbor/math.hpp>

#include <arbor/morph/error.hpp>
#include <arbor/morph/sample_tree.hpp>

#include "algorithms.hpp"
#include "io/sepval.hpp"
#include "util/span.hpp"

namespace arb {

sample_tree::sample_tree(std::vector<msample> samples, std::vector<size_t> parents) {
    if (samples.size()!=parents.size()) {
        throw std::runtime_error(
            "The same number of samples and parent indices used to create a sample morphology");
    }
    reserve(samples.size());
    for (auto i: util::make_span(samples.size())) {
        append(parents[i], samples[i]);
    }
}

void sample_tree::reserve(size_t n) {
    samples_.reserve(n);
    parents_.reserve(n);
    props_.reserve(n);
    branch_ids_.reserve(n);
}

size_t sample_tree::append(size_t p, const msample& s) {
    if (p>size()) {
        throw morphology_error("Parent id of a sample must be less than the sample id");
    }
    auto id = size();

    samples_.push_back(s);
    parents_.push_back(p);

    // Set the point properties for the new point, and update those of its parent as needed.
    point_prop prop = point_prop_mask_none;
    if (!id) {
        // if adding the first sample mark it as root
        set_root(prop);
        single_root_tag_ = true;
    }
    else {
        // Mark the new node as terminal, and unset the parent sample's terminal bit.
        set_terminal(prop);
        const bool term_parent = is_terminal(props_[p]); // track if the parent was terminal.
        unset_terminal(props_[p]);

        // Mark if the new sample is collocated with its parent.
        if (is_collocated(s, samples_[p])) {
            set_collocated(prop);
        }

        // Set parent to be a fork if it was not a terminal point before the
        // new sample was added (and if it isn't the root).
        if (p && !term_parent) {
            set_fork(props_[p]);
        }

        // If the root is the parent and it has the same tag, record that
        // one of the root shares its tag with one of its children.
        if (!p && s.tag==samples_[0].tag) {
            single_root_tag_ = false;
        }
    }
    props_.push_back(prop);

    return id;
}

size_t sample_tree::append(size_t p, const std::vector<msample>& slist) {
    if (!slist.size()) return size();

    for (auto& s: slist) {
        p = append(p, s);
    }

    return p;
}

std::size_t sample_tree::size() const {
    return samples_.size();
}

bool sample_tree::single_root_tag() const {
    return single_root_tag_;
}

const std::vector<msample>& sample_tree::samples() const {
    return samples_;
}

const std::vector<size_t>& sample_tree::parents() const {
    return parents_;
}

const std::vector<point_prop>& sample_tree::properties() const {
    return props_;
}

std::ostream& operator<<(std::ostream& o, const sample_tree& m) {
    o << "sample_tree:"
      << "\n  " << m.size() << " samples"
      << "\n  samples [" << io::csv(m.samples_) <<  "]"
      << "\n  parents [" << io::csv(m.parents_) <<  "]";
    return o;
}

sample_tree swc_as_sample_tree(const std::vector<swc_record>& swc_records) {
    sample_tree m;
    m.reserve(swc_records.size());

    for (auto i: util::count_along(swc_records)) {
        auto& r = swc_records[i];
        // The parent of soma must be 0, while in SWC files is -1
        size_t p = i==0? 0: r.parent_id;
        m.append(p, msample{{r.x, r.y, r.z, r.r}, r.tag});
    }
    return m;
}

} // namespace arb
