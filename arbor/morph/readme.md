# Morphology

This path contains implementation of the *morphology* code that is responsible for
* Building morphologies represented as a set of connected sample points.
* Parsing the morphology tree.
* `TODO` describing regions and points on morphology abstractly.
* `TODO` building a concrete set of points of unbranched cable segments from applying a description to a morphology.

## Describing morphologies as samples

Arbor uses a sample-based approach of constructing and describing cell morphologies that is compatible with the widely-used SWC and NeuroML standards.
The basic unit used to describe a morphology is a sample, which is a tuple (`x`,`y`,`z`,`radius`,`tag`).

Some definitions:
* **root point** = the first sample. Has index 0.
* **branch point** or fork point = a point with more than one child point
* **terminal point** = a point with no children
* **contiguous points** = points with direct parent-child connection
* **collocated** = points that are collocated have the same (x,y,z) location in space, though no neccesarily the same radius.


Samples also satisfy the following
1. two contiguous points can be collocated to indicate a discontinuity in the cable radius at that location.
2. three contiguous points can't be collocated
3. terminal points can't be collocated

Above, points 2 and 3 above are because they likely indicate user error.

### sample indexes

Samples are stored in a vector

* Samples are indexed with non-negative id in the range `[0, num_samples)`
* The first sample, with id 0, is the **root sample**.
* TODO Define parent index
* Each non-root sample must have a parent sample with id less than itself.
    * The root sample is its own parent.
* all samples in an unbranched cable are contiguous. [TODO: needs definition of unbranched cable].

### sample tags

* Each sample has a non-negative integer tag.
* The tag corresponds to SWC kind.
# The tag of the root sample is called the **root tag**

### interpretation of sample sets

Branches can be either

* Spherical, with a center and radius.
* Cables, composed of a sequence of samples, of which at least two are not collocated.

* All samples with the root tag must be either the root sample, or have a parent with the same tag
    * As a consequence the region with root tag is compact, and includes the root sample
* If the root sample is the only sample with root tag it is treated as the center of a spherical branch
* Only the root sample can be used to construct a spherical branch
* joins between branches that attach to a sphere have to be special-cased.
