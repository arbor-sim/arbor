#pragma once

#include <cassert>
#include <functional>
#include <vector>
#include <string>

#include <arbor/swcio.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

/// Morphology composed of samples.
class sample_tree {
    std::vector<msample> samples_;
    std::vector<msize_t> parents_;
    std::vector<point_prop> props_;

public:
    sample_tree() = default;
    sample_tree(std::vector<msample>, std::vector<msize_t>);

    // Reserve space for n samples.
    void reserve(msize_t n);

    // The append functions return a handle to the last sample appended by the call.

    // Append a single sample.
    msize_t append(msize_t p, const msample& s); // to sample p.
    msize_t append(const msample& s); // to the last sample in the tree.

    // Append a sequence of samples.
    msize_t append(msize_t p, const std::vector<msample>& slist); // to sample p.
    msize_t append(const std::vector<msample>& slist); // to the last sample in the tree.

    // The number of samples in the tree.
    msize_t size() const;
    bool empty() const;

    // The samples in the tree.
    const std::vector<msample>& samples() const;

    // The parent index of the samples.
    const std::vector<msize_t>& parents() const;

    // The properties of the samples.
    const std::vector<point_prop>& properties() const;

    friend std::ostream& operator<<(std::ostream&, const sample_tree&);
};

/// Build a sample tree from a sequence of swc records.
sample_tree swc_as_sample_tree(const std::vector<swc_record>& swc_records);

} // namesapce arb

