.. _morphology:

.. |br| raw:: html

  <br/>

Morphology
==========

A cell's *morphology* describes it's three-dimensional branching structure.
This guide will describe how morphologies are defined and used in Arbor,
followed by API documentation.

Sample Trees
------------

A *sample tree* is a low-level description of a cell's morphology.
It is simple and flexible to support the diverse descriptions
of cell morphologies (e.g. SWC, NeuroLicida, NeuroML), and to support tools that
iteratively construct cell morphologies (e.g. L-system generators, interactive cell-builders).

The basic unit used to define a morphology in a sample tree is a *sample*, which
is a three-dimensionsal *location*, with a *radius* and *tag* meta-data.
Samples are stored in :class:`msample` types, for example in Python a sample centred at the origin
with radius 3 μm, and a *tag* of 1 can is created with:

.. code:: Python

    s = arbor.msample(x=0, y=0, z=0, radius=3, tag=1)

.. note::

    A *tag* is an integer label on every sample, which can be used later to define
    regions on cell models. For example, tags could store the *structure identifier* field in the
    `SWC format <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_,
    which identifies whether samples lie in the soma, axons, dendrites, etc. In Arbor tag definitions
    are not fixed, and users can customise them for their requirements.

A *sample tree* defines parent-child relationships between samples.
It is assumed that neuron morphology can be modelled as a *tree*, that is, starting with a single
root location, the cables that represent dendrites and axons can branch, but branches can not rejoin.
Given this assumption, it is possible to represent the tree structure by asigning a parent id
to each sample in a list of samples that represent the radius of the cell morphology at a set
of locations.
In a sample tree with *n* samples, each sample has a unique id in the range ``[0, 1, ..., n-1]``.
The following terms are used to label samples in a sample tree:

* *root*: The first sample in the sample tree, with index 0.
* *parent*: Each sample has one parent.
* *child*: The children of sample *s* are samples whose parent is *s*.
* *terminal*: A sample with no children.
* *fork*: A sample with more than one child. Fork points are where a cable splits into two or more branches.

To demonstrate these concepts, the following observations apply to the sample tree :ref:`below <morph-img-stree>`:

* The tree is composed of 7 samples, numbered from 0 to 6 inclusive.
* Sample 3 is a fork point whose children are samples 4 and 6.
* Samples 5 and 6 are terminals, and have no children.
* Every sample has one parent, except for the root sample.

.. _morph-img-stree:

.. figure:: gen-images/stree.svg
  :width: 400
  :align: left
  :alt: A sample tree with 7 points, with root, fork and terminal samples marked.

  A sample tree with 7 samples, and a single fork point. The parent index for
  the sample tree is ``[npos 0 1 2 3 4 3]``, where ``npos`` is a placeholder
  parent index for the root sample.

The following rules and corollaries apply to sample trees in Arbor:

* Every sample has one and only one parent:

  * the root sample has a special placeholder parent indicated with ``npos``.

* In a sample tree with *n* samples, the samples have unique ids in the half open interval *[0, n)*:

  * The root sample has index 0;
  * Every sample has an id greater than the id of its parent;
  * Ids of samples on the same unbranched section do not need to be contiguous.

* A child can be *collocated* with its parent, where both have the same location.
  This is used in practice to indicate a discontinuity in the radius of a cable, or the
  start of a child branch with a different radius than its parent.

Sample Tree Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cell morphologies are constructed from a series of connected truncated frustums, with an optional
spherical segment at the root of the morphology, which conventionally corresponds to the soma.


A sample tree is a connected set of samples, without the concept of truncated frustums, or spherical segments.
A :ref:`morphology <morph-morphology>`

.. _morph-tree1:

Tree₁: Spherical cell
""""""""""""""""""""""""""""""

The simplest sample tree is a single sample, used to represent a *spherical cell*.
This example has a sphere of radius 3 μm centered at the origin.

.. csv-table::
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,        3,     1


.. figure:: gen-images/tree1.svg
  :width: 100
  :align: center

.. _morph-tree2:

Tree₂: Single cable segment
""""""""""""""""""""""""""""""

Next in order of complexity is a single *unbranched cable segment* defined by two samples,
specifically a cable of length 10 μm, with radii of 0.5 μm and 0.25 μm.

.. csv-table::
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,    0.50,     1
      0,      10,   0,   0,    0.25,     1

.. figure:: gen-images/tree2.svg
  :width: 300
  :align: center

.. _morph-tree3:

Tree₃: Y-shaped cell
""""""""""""""""""""""""""""""

The simplest branching morphology is a cable that bifurcates, a *y-shaped cell*.
First branch of the tree is a cable of length 10 μm and radius 0.5 μm.
The two child branches start with the to the distal sample of the first branch
and end with points 0.25 μm.

.. csv-table::
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,    0.50,     1
      0,      10,   0,   0,    0.50,     1
      1,      15,   3,   0,    0.25,     1
      1,      15,  -3,   0,    0.25,     1

.. figure:: gen-images/tree3.svg
  :width: 400
  :align: center

.. _morph-tree4:

Tree₄: Y-shaped cell variant
""""""""""""""""""""""""""""""

For the child branches in Tree₃  to have constant radius of 0.25 μm, instead of tapering from 0.5 μm to 0.25 μm,
additional samples with radius 0.25 μm can be inserted at the start of each branch, collocated with
the distal sample of the first branch.

.. csv-table::
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,    0.50,     1
      0,      10,   0,   0,    0.50,     1
      1,      10,   0,   0,    0.25,     1
      2,      15,   3,   0,    0.25,     1
      1,      10,   0,   0,    0.25,     1
      4,      15,  -3,   0,    0.25,     1

.. figure:: gen-images/tree4a.svg
  :width: 400
  :align: center

.. figure:: gen-images/tree4b.svg
  :width: 400
  :align: center

.. _morph-tree5:

Tree₅: ball and stick
""""""""""""""""""""""""""""""

The next example is a spherical soma of radius 3 μm with a branch of length
7 μm and constant radius of 1 μm attached.

.. csv-table::
   :widths: 10, 10, 10, 10, 10, 10

   *parent*, *x*, *y*, *z*, *radius*, *tag*
   npos,       0,   0,   0,       3,     1
      0,       3,   0,   0,       1,     1
      1,      10,   0,   0,       1,     1

.. figure:: gen-images/tree5.svg
  :width: 300
  :align: center

.. note::
    The sample tree above could
    also be used to represent a single unbranched cable, with two segments:
    the first 3 μm segment tapers from 3 μm to 1 μm, and the second 7 μm segment
    has constant radius of 1 μm.

    This ambiguity is cleared up when
    :ref:`transforming a sample tree into a morphology <morph-morphology>`.

Sample Tree Construction
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _morph-morphology:

Morphology
----------

A sample tree does not describe the actual branches

Arbor treats morphologies as a tree of truncated frustums, with an optional spherical segment at the root of the tree.

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

For morphologies with a spherical root, the root sample defines a special spherical branch.
Rules imposed on spherical:
* samples with parent *root* (i.e. *s.parent==0*) must be a distance of *root.radius* from *root.location*.
* samples with *root* as parent are the start of a branch, and hence must have at least one child sample.
