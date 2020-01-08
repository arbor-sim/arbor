#include <iostream>
#include <stdexcept>
#include <vector>

#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/sample_tree.hpp>

#include "io/sepval.hpp"
#include "util/span.hpp"
#include "util/transform.hpp"

namespace arb {

sample_tree::sample_tree(std::vector<msample> samples, std::vector<msize_t> parents) {
    if (samples.size()!=parents.size()) {
        throw std::invalid_argument("sample and parent vectors differ in size");
    }
    reserve(samples.size());
    for (auto i: util::make_span(samples.size())) {
        append(parents[i], samples[i]);
    }
}

void sample_tree::reserve(msize_t n) {
    samples_.reserve(n);
    parents_.reserve(n);
    props_.reserve(n);
}

msize_t sample_tree::append(msize_t p, const msample& s) {
    if ((empty() && p!=mnpos) || p>=size()) {
        if (p!=mnpos) throw invalid_sample_parent(p, size());
    }

    auto id = size();
    samples_.push_back(s);
    parents_.push_back(p);

    // Set the point properties for the new point, and update those of its parent as needed.
    point_prop prop = point_prop_mask_none;
    if (!id) {
        // if adding the first sample mark it as root
        set_root(prop);
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
    }
    props_.push_back(prop);

    return id;
}

msize_t sample_tree::append(const msample& s) {
    return append(empty()? mnpos: size()-1, s);
}

msize_t sample_tree::append(msize_t p, const std::vector<msample>& slist) {
    if (!slist.size()) return size();

    for (auto& s: slist) {
        p = append(p, s);
    }

    return p;
}

msize_t sample_tree::append(const std::vector<msample>& slist) {
    return append(empty()? mnpos: size()-1, slist);
}

msize_t sample_tree::size() const {
    return samples_.size();
}

bool sample_tree::empty() const {
    return samples_.empty();
}

const std::vector<msample>& sample_tree::samples() const {
    return samples_;
}

const std::vector<msize_t>& sample_tree::parents() const {
    return parents_;
}

const std::vector<point_prop>& sample_tree::properties() const {
    return props_;
}

std::ostream& operator<<(std::ostream& o, const sample_tree& m) {
    auto tstr = util::transform_view(m.parents_,
            [](msize_t i) -> std::string {
                return i==mnpos? "npos": std::to_string(i);
            });
    return o << "(sample_tree (\n  " << io::sepval(m.samples_, "\n  ") <<  ")\n"
             << "  (" << io::sepval(tstr, ' ') <<  "))";
}

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

} // namespace arb
