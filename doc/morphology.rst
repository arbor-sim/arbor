.. _morphology:

Morphology
==========

In Arbor, a *morphology* describes a cell's three-dimensional branching structure.
Arbor treats morphologies as a tree of truncated frustums, with an optional spherical segment at the root of the tree.

Sample Trees
------------

A sample is a three-dimensionsal location, with additional radius and tag meta-data:
* ``x``, ``y``, ``z`` : *location* of the sample.
* ``radius``: the *radius* of the cable/sphere centered at *location*.
* ``tag``: a non-negative integer for encoding additional user-defined meta data.

**CODE** example of creating a sample.

A *sample tree* defines parent-child relationships between samples that define a morphology.
It is assumed that neuron morphology can be modelled as a *tree*, that is, starting with a single
root sample, the cables that represent dendrites and axons can branches, but branches can not
rejoin.
Given this assumption, it is possible to represent the tree structure by augmenting a set of
samples with a parent index for each sample.

* *root*: The first sample in the sample tree, with index 0.
* *parent*: Each sample has one and only one parent sample.
* *child*: The children of sample *s* are samples whose parent is *s*.
* *cable segment*: an unbranched cable between two sample points *s₁* and *s₂*.
  A segment is a tapered cylinder, with radius at each end 

The following rules and corollaries apply to the samples in a sample tree:

* Every sample has one and only one parent:

  * the root sample has a special placeholder parent indicated with ``npos``.

* A sample can have 0 or more children.
* In a sample tree with *n* samples, the samples have unique ids in the half open interval *[0, n)*:

  * The root sample has index 0
  * Every sample has an id greater than the id of its parent.
  * Ids of samples on the same unbranched section do not need to be contiguous.

* foobar
* a child can have the same location as its parent, which is used to step discontinuity in radius.

Sample Tree Examples
~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest sample tree, besides the trivial empty tree, is a single sample, which is used
to represent a spherical cell.

.. csv-table:: Example 1: single sample
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,        3,     1

Next in order of complexity is a single unbranched cable segment between two samples.
In this case we have a cable of length 10 μm, with radii of 0.5 μm and 0.25 μm.

.. csv-table:: A single cable segment
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,    0.50,     1
      0,       0,   0,  10,    0.25,     1

.. csv-table:: A y-shaped cell
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,    0.50,     1
      0,       0,   0,  10,    0.25,     1
      1,       0,   5,  15,    0.25,     1
      1,       0,   5,  15,    0.25,     1

Sample Tree Construction
~~~~~~~~~~~~~~~~~~~~~~~~~

Sample trees are built by appending samples to an initially empty tree.

.. csv-table:: two point morpho
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,        3,     1
      0,       0,   0,   0,        3,     1


Morphology
----------

A morphology is a concrete description of a morphology in terms of cable branches, and optionally a spherical root branch.

When spherical roots are optional, as in Arbor, it is possible for the morphology represented by a sample tree to be ambiguous.

* *fork*: a sample that has at least two children.
* *terminal*: a sample that has no children (e.g. the tip of a dendrite.)
* *cable branch*: an unbranched series of cable segments that has fork points at each end.
* *spherical branch*: always at root, represented by a single sample, sphere with center *location* and *radius*.
* *distal*: away from the root relative.
* *proximal*: towards the root relative.

rules:

* Every cable branch has at least two samples, one at either end, which are referred to as the proximal and distal samples of the branch.

When a morphology spherical root, the root sample defines a special spherical branch.
Rules imposed on spherical:
* samples with parent *root* (i.e. *s.parent==0*) must be a distance of *root.radius* from *root.location*.
* samples with *root* as parent are the start of a branch, and hence must have at least one child sample.
