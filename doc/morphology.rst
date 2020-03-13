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
* *collocated*: Two points are collocated if they have the same location.

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
  More than two samples can be collocated at the same location, however there must be a direct
  parent-child relationship between all samples.
  This is used in practice to indicate a discontinuity in the radius of a cable, or the
  start of a child branch with a different radius than its parent.

This sample tree can be created in Python by creating an empty sample tree, and appending samples to the tree:

.. code:: Python

    import arbor
    tree = arbor.sample_tree()
    tree.append(          x= 0.0, y= 0.0, z=0.0, radius=3.0, tag=1)
    tree.append(parent=0, x= 5.0, y=-1.0, z=0.0, radius=1.2, tag=1)
    tree.append(parent=1, x=10.0, y= 0.5, z=0.0, radius=1.2, tag=3)
    tree.append(parent=2, x=15.0, y= 0.0, z=0.0, radius=1.0, tag=3)
    tree.append(parent=3, x=18.0, y= 5.0, z=0.0, radius=1.0, tag=3)
    tree.append(parent=4, x=23.0, y= 8.0, z=0.0, radius=0.7, tag=3)
    tree.append(parent=3, x=20.0, y=-4.0, z=0.0, radius=0.8, tag=3)

A ``parent`` isn't provided when adding the root sample, and can dropped when adding subsequent samples
if the sample's parent is the last sample that was added to the tree.
Sample trees constructed in this manner, where the parent of each new sample must already be in the tree
are always valid.

.. _morph-morphology:

Morphology
----------

Sample trees do not describe the geometry between samples or whether the sample
at the root of the tree should be interpreted as a sphere or as the start of one or more cable sections.
This interpretation is provided in Arbor by a *morphology*, which interpr takes a sample tree 

The following terms are used to describe parts of a cell morphology, on top of those introduced above
for sample trees:

* *cable segment*: a frustum (cylinder or truncated cone) between two adjacent samples.
* *branch*: an unbranched sequence of cable segments that has fork, terminal or root points at each end.
* *spherical branch*: a special branch, represented by a single sample, that is a sphere with center and radius defined by the root sample.
* *distal*: a location is distal to another if it is further from the root.
* *proximal*: a location is distal to another if it is further from the root.
* *location*: a point on the morphology, uniquely identified by a tuple ``(branch, pos)``, where branch identifies the branch, and pos is a relative position  between 0 and 1.

Taking the example sample tree from above:

.. code:: Python

    >>> morph = arbor.morphology(tree, spherical_soma=False)
    # query basic information about the morphology
    >>> morph.num_branches
    3
    >>> morph.num_samples
    7
    >>> morph.spherical_root
    False

    # query the sample indexes along each branch
    >>> morph.branch_indexes(0)
    [0, 1, 2, 3]
    >>> morph.branch_indexes(1)
    [3, 4, 5]
    >>> morph.branch_indexes(2)
    [3, 6]

    # The ids of the branches that are children of branch 0
    >>> morph.branch_children(0)
    [1, 2]
    # The ids of the parent of branch 1
    >>> morph.branch_parent(0)
    0

    # the underlying samples and parents
    >>> morph.samples
    [(sample (point 0 0 0 3) 1), (sample (point 5 -1 0 1.2) 3), (sample (point 10 0.5 0 1.2) 3), (sample (point 15 0 0 1) 3), (sample (point 18 5 0 1) 3), (sample (point 23 8 0 0.7) 3), (sample (point 20 -4 0 0.8) 3)]
    >>> morph.sample_parents
    [4294967295, 0, 1, 2, 3, 4, 3]


Rules about branches

* Every cable branch has at least two samples, which define a single cable segment.
* There can be either zero or one spherical 
* Branches are numbered starting from 0

    * parent id less than child id
    * root branch has index 0, and always contains the root sample.
    * branches are numbered deterministically from a sample tree: in the order of the first sample in each branch.

For morphologies with a spherical root, the root sample defines a special spherical branch.
Rules imposed on spherical:

* samples with parent *root* (i.e. *s.parent==0*) must be a distance of *root.radius* from *root.location*.
* samples with *root* as parent are the start of a branch, and hence must have at least one child sample.

Morphology Construction
~~~~~~~~~~~~~~~~~~~~~~~~~

In Arbor morphologies can have the following features:

* Spherical soma.
* Somas compsed of cylindrical segments.
* Cables that taper linearly between two points.
* Cables that have step discontinuities in radius.
* Branches that bifurcate into branches with different radius.

These features are required to faithfully reproduce morphologies specified in
model specification formats such as SWC, NeuroML, NEURON HOC and NeuroLicida ASCI ASC files.
Methods for reproducing these features are shown in a seris of examples below.

In all of the exampels morphologies are two dimensional, with the z-dimension set to zero,
for illustration.

.. _morph-tree1:

Example 1: Spherical cell
""""""""""""""""""""""""""""""

The simplest morphology in Arbor is a sphere.
For this a  single sample, used to represent a *spherical cell*.
This example has a sphere of radius 3 μm centered at the origin.

.. csv-table::
   :widths: 8, 10, 10, 10, 10, 10, 10

   *id*, *parent*, *x*, *y*, *z*, *radius*, *tag*
   0,    npos,       0,   0,   0,        3,     1


.. figure:: gen-images/tree1.svg
  :width: 100
  :align: center

  The sample tree is a single sample with radius 3 μm.

.. figure:: gen-images/morph1.svg
  :width: 100
  :align: center

  The corresponding morphology is a sphere of radius 3 μm.

.. _morph-tree2:

Example 2: Unbranched cable
""""""""""""""""""""""""""""""

Next in order of complexity is a cable branch, with no fork points.

We start with a cable of length 10 μm, with a radius that tapes from 0.5 μm to 0.25 μm
at the proximal and distal ends respectively.
It is constructed from a sample tree of two points that define the end points of the cable.

.. csv-table::
   :widths: 8, 10, 10, 10, 10, 10, 10

   *id*, *parent*, *x*, *y*, *z*, *radius*, *tag*
   0,    npos,       0,   0,   0,    0.50,     1
   1,       0,      10,   0,   0,    0.25,     1

.. figure:: gen-images/tree2a.svg
  :width: 300
  :align: center

  The sample tree has a root and terminal point, marked in blue and green respectively.

.. figure:: gen-images/morph2a.svg
  :width: 300
  :align: center

  The morphology is a tapered cable with one cable segment.

The radius and centre of a cable segment vary lineary between its end points. To define an unbranched cable
with non uniform center and/or radius, use multiple samples to build a piecewise linear reconstruction
of the cable geometry.
This example starts and ends at the same locations as the previous, however it is constructed from 4
distinct cable segments:

.. csv-table::
   :widths: 8, 10, 10, 10, 10, 10, 10

   *id*, *parent*, *x*, *y*, *z*, *radius*, *tag*
   0,     npos,  0.0,  0.0,  0.0,  1.0,    1
   1,        0,  3.0,  0.2,  0.0,  0.8,    1
   2,        1,  5.0, -0.1,  0.0,  0.7,    1
   3,        2,  8.0,  0.0,  0.0,  0.6,    1
   4,        3, 10.0,  0.0,  0.0,  0.5,    1

.. figure:: gen-images/tree2b.svg
  :width: 300
  :align: center

  The sample tree has 5 samples, where each sample has at most one child.

.. figure:: gen-images/morph2b.svg
  :width: 600
  :align: center

  **left:** The resulting morphology is an ubranched cable comprised of 4 cable segments.

  **right** The four segments form a one contiguous branch.

Collocated samples can be used to create a discontinuity in cable radius.
The next example adds a discontinuity to the previous example at sample 2, where the
radius now changes from 0.7 μm to 0.3 μm:

.. csv-table::
   :widths: 8, 10, 10, 10, 10, 10, 10

   *id*, *parent*, *x*, *y*, *z*, *radius*, *tag*
   0,     npos,  0.0,  0.0,  0.0,  1.0,    1
   1,        0,  3.0,  0.2,  0.0,  0.8,    1
   2,        1,  5.0, -0.1,  0.0,  0.7,    1
   3,        2,  5.0, -0.1,  0.0,  0.3,    1
   4,        3,  8.0,  0.0,  0.0,  0.5,    1
   5,        4, 10.0,  0.0,  0.0,  0.5,    1

.. figure:: gen-images/tree2c.svg
  :width: 300
  :align: center

  Samples 2 and 3 are collocated with different radii.

.. figure:: gen-images/morph2c.svg
  :width: 600
  :align: center

  The resulting morphology has a step discontinuity in radius.

.. _morph-tree3:

Example 3: Y-shaped cell
""""""""""""""""""""""""""""""

The simplest branching morphology is a cable that bifurcates, a *y-shaped cell*.
First branch of the tree is a cable of length 10 μm and radius 0.5 μm.
The two child branches start with the to the distal sample of the first branch
and end with points 0.25 μm.

.. csv-table::
   :widths: 8, 10, 10, 10, 10, 10, 10

   *id*, *parent*, *x*, *y*, *z*, *radius*, *tag*
   0,    npos,       0,   0,   0,    0.50,     1
   1,       0,      10,   0,   0,    0.50,     1
   2,       1,      15,   3,   0,    0.25,     1
   3,       1,      15,  -3,   0,    0.25,     1

.. figure:: gen-images/tree3.svg
  :width: 400
  :align: center

.. figure:: gen-images/morph3.svg
  :width: 400
  :align: center

The child branches above start with the same radius of 0.5 μm as the distal end of their parent branch.
If we wanted the branches to have a constant radius of 0.25 μm, instead of tapering from 0.5 μm to 0.25 μm,
we use collocated samples of radius 0.25 μm.
Two methods that use the same approach are illustrated below:
* insert collocated points at the start of each child branch;
* insert a single collocated point at the end of the parent branch.

.. csv-table::
   :widths: 8, 10, 10, 10, 10, 10, 10

   *id*, *parent*, *x*, *y*, *z*, *radius*, *tag*
   0,     npos,  0.0,  0.0,  0.0,  1.0,    1
   1,        0, 10.0,  0.0,  0.0,  0.5,    1
   2,        1, 10.0,  0.0,  0.0,  0.2,    1
   3,        2, 15.0,  3.0,  0.0,  0.2,    1
   4,        1, 10.0,  0.0,  0.0,  0.2,    1
   5,        4, 15.0, -3.0,  0.0,  0.2,    1

   *id*, *parent*, *x*, *y*, *z*, *radius*, *tag*
   0,     npos,  0.0,  0.0,  0.0,  1.0,    1
   1,        0, 10.0,  0.0,  0.0,  0.5,    1
   2,        1, 10.0,  0.0,  0.0,  0.2,    1
   3,        2, 15.0,  3.0,  0.0,  0.2,    1
   4,        2, 15.0, -3.0,  0.0,  0.2,    1

.. figure:: gen-images/tree4a.svg
  :width: 400
  :align: center

  The first approach has 3 collocated points at the fork: sample 1 is at the end
  of the parent branch, and samples 2 and 4 are attached to sample 1 and are at
  the start of the children branches.

.. figure:: gen-images/tree4b.svg
  :width: 400
  :align: center

  The second approach has 2 collocated points at the fork, and the children attach to 
  the second collocated sample, which sets the radius to 0.25 μm.

.. figure:: gen-images/morph4.svg
  :width: 400
  :align: center

  The resulting morphology is the same for both approaches.

.. _morph-tree5:

Tree₅: ball and stick
""""""""""""""""""""""""""""""

The next example is a spherical soma of radius 3 μm with a branch of length
7 μm and constant radius of 1 μm attached.

.. csv-table::
   :widths: 8, 10, 10, 10, 10, 10, 10

   *id*, *parent*, *x*, *y*, *z*, *radius*, *tag*
   0,     npos,  0.0,  0.0,  0.0,  2.0,    1
   1,        0,  2.0,  0.0,  0.0,  1.0,    1
   2,        1, 10.0,  0.0,  0.0,  1.0,    1

.. figure:: gen-images/tree5.svg
  :width: 300
  :align: center

This sample tree can be interpreted two ways: either as a single unbranched cable composed of
two cable segments, or as two branches: a spherical root with a cable branch attached:

.. figure:: gen-images/morph5a.svg
  :width: 600
  :align: center

.. figure:: gen-images/morph5b.svg
  :width: 600
  :align: center

Example 6: a bit more involved
""""""""""""""""""""""""""""""

Take our example from throughout this page:

.. figure:: gen-images/tree6a.svg
  :width: 400
  :align: center

.. figure:: gen-images/morph6a.svg
  :width: 800
  :align: center

.. figure:: gen-images/morph6c.svg
  :width: 800
  :align: center

To use a spherical soma, add an aditional sample on the edge of the soma that represents the
start of the dendrite that branches off the soma, then instantiate the morphology with
``spherical_root`` set to ``True``:

.. figure:: gen-images/tree6b.svg
  :width: 400
  :align: center

.. figure:: gen-images/morph6b.svg
  :width: 800
  :align: center

