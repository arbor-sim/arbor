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
* ``tag``: an integer for encoding additional user-defined meta data.

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

The simplest sample tree is a single sample, used to represent a *spherical cell*.
:ref:`Tree₁ <morph-tree1>` describes a sphere of radius 3 μm centered at the origin.

.. _morph-tree1:

.. csv-table:: Tree₁: Single sample tree
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,        3,     1

Next in order of complexity is a single *unbranched cable segment* defined by two samples.
:ref:`Tree₂ <morph-tree2>` defines a cable of length 10 μm, with radii of 0.5 μm and 0.25 μm.

.. _morph-tree2:

.. csv-table:: Tree₂: Single cable segment
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,    0.50,     1
      0,       0,   0,  10,    0.25,     1

The simplest branching morphology is a cable that bifurcates, a *y-shaped cell*.
First branch of the tree is a cable of length 10 μm and radius 0.5 μm.
The two child branches start with the to the distal sample of the first branch
and end with points 0.25 μm.

.. _morph-tree3:

.. csv-table:: Tree₃: y-shaped cell
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,    0.50,     1
      0,       0,   0,  10,    0.50,     1
      1,       0,   3,  15,    0.25,     1
      1,       0,  -3,  15,    0.25,     1


For the child branches in Tree₃  to have constant radius of 0.25 μm, instead of tapering from 0.5 μm to 0.25 μm,
additional samples with radius 0.25 μm can be inserted at the start of each branch, collocated with
the distal sample of the first branch.

.. _morph-tree4:

.. csv-table:: Tree₄: y-shaped cell
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,    0.50,     1
      0,       0,   0,  10,    0.50,     1
      1,       0,   0,  10,    0.25,     1
      2,       0,   3,  15,    0.25,     1
      1,       0,   0,  10,    0.25,     1
      4,       0,  -3,  15,    0.25,     1

The next example is a spherical soma of radius 3 μm with a branch of length
7 μm and constant radius of 1 μm attached.

.. note::
    This sample tree could
    also be used to represent a single unbranched cable, with two segments:
    the first 3 μm segment tapers from 3 μm to 1 μm, and the second 7 μm segment
    has constant radius of 1 μm.

    This ambiguity is cleared up when
    :ref:`transforming a sample tree into a morphology <morph-morphology>`.

.. _morph-tree5:

.. csv-table:: Tree₅: ball and stick
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,       3,     1
      0,       0,   0,   3,       1,     1
      1,       0,   0,  10,       1,     1

Sample Tree Construction
~~~~~~~~~~~~~~~~~~~~~~~~~


.. _morph-morphology:

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
