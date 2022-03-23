#pragma once

#include <optional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/region.hpp>

namespace arb {

// Stiches represent an alternative building block for morphologies.
//
// A stitch describes a portion of the morphology delimited by two
// `mpoint`s. Stitches can be attached to a parent stich at any
// point along the parent, interpolated linearly from the end points.
// Each stitch is associated with a unique string label, and optionally
// an integer tag value.
//
// The stitch builder collects stitches and produces the corresponding
// morphology and region/location labels.

struct mstitch {
    std::string id;
    std::optional<mpoint> prox;
    mpoint dist;
    int tag;

    mstitch(std::string id, mpoint prox, mpoint dist, int tag = 0):
        id(std::move(id)), prox(std::move(prox)), dist(std::move(dist)), tag(tag)
    {}

    mstitch(std::string id, mpoint dist, int tag = 0):
        id(std::move(id)), dist(std::move(dist)), tag(tag)
    {}
};

struct stitch_builder_impl;
struct stitched_morphology;

struct ARB_ARBOR_API stitch_builder {
    stitch_builder();

    stitch_builder(const stitch_builder&) = delete;
    stitch_builder(stitch_builder&&);

    stitch_builder& operator=(const stitch_builder&) = delete;
    stitch_builder& operator=(stitch_builder&&);

    // Make a new stitch in the morphology, return reference to self.
    //
    // If the stitch does not contained a proximal point, it will be
    // inferred from the point where it attaches to the parent stitch.
    // If the parent is omitted, it will be taken to be the last stitch
    // added.

    stitch_builder& add(mstitch f, const std::string& parent_id, double along = 1.);
    stitch_builder& add(mstitch f, double along = 1.);

    ~stitch_builder();
private:
    friend stitched_morphology;
    std::unique_ptr<stitch_builder_impl> impl_;
};

// From stitch builder construct morphology, region expressions.

struct stitched_morphology_impl;

struct ARB_ARBOR_API stitched_morphology {
    stitched_morphology() = delete;
    stitched_morphology(const stitch_builder&); // implicit
    stitched_morphology(stitch_builder&&); // implicit

    stitched_morphology(const stitched_morphology&) = delete;
    stitched_morphology(stitched_morphology&&);

    arb::morphology morphology() const;
    region stitch(const std::string& id) const;
    std::vector<msize_t> segments(const std::string& id) const;

    // Create labeled regions for each stitch with label equal to the stitch id, prepended by `prefix`.
    label_dict labels(const std::string& prefix="") const;

    ~stitched_morphology();
private:
    std::unique_ptr<stitched_morphology_impl> impl_;
};

} // namesapce arb
