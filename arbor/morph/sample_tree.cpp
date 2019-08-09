#include <cmath>
#include <vector>

#include <arbor/math.hpp>

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
}

size_t sample_tree::append(size_t p, const msample& s) {
    if (p>size()) {
        throw std::runtime_error("parent id of a sample must be less than the sample id");
    }
    samples_.push_back(s);
    parents_.push_back(p);

    return size()-1;
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

const std::vector<msample>& sample_tree::samples() const {
    return samples_;
}

const std::vector<size_t>& sample_tree::parents() const {
    return parents_;
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
