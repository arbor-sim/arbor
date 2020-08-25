.. _morphology:

Morphology
==========

A cell's *morphology* describes both its geometry and branching structure.
Morphologies in Arbor are modeled as a set of one dimensional cables of variable radius,
joined together to form a tree. The cables are described by a series of segments,
which are truncated conic frustums.

Segment Trees
--------------

A *segment tree* is a description of a the segments and their connections
that is designed to support both the diverse descriptions
of cell morphologies (e.g. SWC, NeuroLicida, NeuroML), and tools that
iteratively construct cell morphologies (e.g. L-system generators, interactive cell-builders).

The building blocks of a segment tree are *points* and *segments*:

* *point*: a three-dimensional location and a radius, used to mark the centre and radius
  of the cable.

* *segment*: a frustum (cylinder or truncated cone), with the centre and radius at each
  end defined by a pair of points.

.. csv-table:: Fields that define a point.
   :widths: 10, 10, 30

   **Field**,   **Type**, **Description**
   ``x``,       real, x coordinate of centre of cable (μm).
   ``y``,       real, y coordinate of centre of cable (μm).
   ``z``,       real, z coordinate of centre of cable (μm).
   ``radius``,  real, cross sectional radius of cable (μm).

.. csv-table:: Fields that define a segment.
   :widths: 10, 10, 30

   **Field**,   **Type**, **Description**
   ``prox``,       point,   the center and radius of the proximal end.
   ``dist``,       point,   the center and radius of the distal end.
   ``tag``,        integer, tag meta-data.

.. _morph-tags:

Tags
~~~~~~~~~~~~~~~~~~

A *tag* is an integer label on every segment, which can be used to define disjoint
regions on cells, on which dynamics or electrical properties can be specified.

When working with morphologies loaded from
`SWC files <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_,
the tag can be derived from the *structure identifier* field used to indicate
whether individual SWC records lie in the soma, axons, dendrites, etc.

However, the meaning of tag values are not fixed in Arbor, and they can be used to
define model-specific regions.

.. _morph-sample-definitions:

Segment Tree Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Segment trees comprise a sequence of segments starting from at lease one *root* segment,
together with a parent-child adjacency relationship where a child segment is
distal to its parent.
Branches in the tree occur where a segment has more than one child.
Furthermore, a segment can not have more than one parent.
In this manner, neuron morphologies are modeled as a *tree*, where cables that
represent dendrites and axons can branch, but branches can not rejoin.

The following definitions are used to refer to segments in a segment tree:

* *root*: segments at the root or start of the tree. A non-empty tree must have at least one root segment,
  and the first segment will always be a root.

* *parent*: Each segment has one parent, except for root segments which have ``mnpos`` as their parent.

  * A segments's id is always greater than the id of its parent.
  * The ids of segments on the same unbranched sequence of segments do not need to be contiguous.

* *child*: The children of segment *s* are segments whose parent is *s*.
* *terminal*: A segment with no children. Terminals lie at the end of dendritic trees or axons.
* *fork*: A segment with more than one child. The distal end of a fork segment are fork points,
  where a cable splits into two or more branches.

  * Arbor allows more than two branches at a fork point.

The folowing segment tree models a soma with a cylinder, a branching dendritic tree and
an axon with an axonal hillock. The segments are colored according to their tag, which
in this case are SWC structure identifiers: tag 1 colored pink for soma;
tag 2 colored grey for axon; tag 3 colored blue for basal dendrites.

.. _morph-label-seg-fig:

.. figure:: gen-images/label_seg.svg
  :width: 600
  :align: center

We can apply the following labels to the segments:

* The tree is composed of 11 segments (1 soma, 2 axon, 8 dendrite).
* The proximal ends of segments 0 and 9 (the soma and axon hillock respectively) are attached to the root of the tree.
* Segment 2 is a fork, with segments 3 and 5 as children.
* Segment 5 is a fork, with segments 7 and 8 as children.
* There is also a fork at the root, whith segments 0 and 9 as children.
* Segments 4, 6, 8 and 10 are terminal segments.

In the example above there are no gaps between segments, however
it is possible for segments to be detached, where the proximal end of a segment is not coincident
with the distal end of its parent. The following morphology has gaps between the start of the
axon and dendrititic tree and the soma segment to which they attach.

.. _morph-detached-seg-fig:

.. figure:: gen-images/detached_seg.svg
  :width: 600
  :align: center

.. note::
    In Arbor, segments are always treated as though they are connected directly
    to their parents, regardless of whether ends where they attached are collocated.

    Gaps are frequently the result of simplifying the the soma,
    whereby the complex geometry of a soma is represented using a cylinder or sphere
    (spheres are represented by a cylinder with length and diameter equal to that of
    the sphere in simulation tools like Arbor and NEURON).

    A gap between a cylindrical soma and segments attached to it does not mean
    that the segmentation is invalid.
    To illustrate why this can occur, consider a potato-shaped soma modeled with a
    cylinder of the same surface area.
    If the cell description places the first segment of a dendritic tree where it attaches to
    the "potato soma", it is unlikely to be collocated with an end of the simplified soma.
    The cell model will correctly represent the location and dimension of the dendritic tree,
    while preserving the soma surface area with a simplified cylindrical model.

Because Arbor supports tapered segments (where radius varies linearly along a segment) it is possible to
represent more complex soma shapes using multiple segments, for example the segmentation below
uses 4 segments to model the soma.

.. _morph-stacked-seg-fig:

.. figure:: gen-images/stacked_seg.svg
  :width: 600
  :align: center

.. _morph-morphology:

Morphology
----------

A *morphology* describes the geometry of a cell as unbranched cables with variable radius
, and their associated tree structure. They are constructed from a segment tree by defining
the branches, which are uniquely derived from the cable segments.

Every *segment tree* can be used to generate a unique *morphology*, which derives and enumerates
*branches* fromt the segments.
The branches of a morphology are unbranched cables, composed of one or more segments, where:

  * the first (proximal) segment of the branch is either a root or the child of fork segment.
  * the last (distal) segment of the branch is either a fork or terminal segment.

Branches are enumerated according to the ids of their proximal segments in the segment trree.


When constructed in this manner, the following statements hold for the generted branches and
their enumeration:

  * Because a branch must have root, fork or terminal ends, a branch can not be sub-divided
    into two or more branches, and hence there is only one possible set of branches that
    can be derived from a segment tree.
  * Because branches are enumerated according to the id of their proximal segments,
    there is only one branch enumeration representation for a segment tree.
  * However, it is possible for two topologically equivalent morphologies to be
    derived from different segment trees (e.g. two trees with the same segments, however
    different valid segment enumerations), and potentially have different branch numbers.
  * Every valid segment tree can be used to construct a valid morphology.

.. Note::

    Because two topologically-equivalent morphologies may have different segment and
    branch numbering, it is important that model descriptions should avoid refering to
    branches or segments by id.
    This should be relaxed only in well-understood situations, for example when working with
    models that always represent to soma with a single segment at the root of the tree,
    which will always have segment id 0.

To illustrate branch generation, consider the first segment tree example on this page,
which is illustrated along with its branches below:

.. _morph-label-morph-fig:

.. figure:: gen-images/label_morph.svg
  :width: 800
  :align: center

The first branch contains the soma and the first two segments of the dendritic tree.
There are four more branches in the dendritic tree, and one representing the two
segments of the axon.

Note, that though it is possible to create an unbranched sequence of segments composed
of the axon, soma and first two segements in the dendritic tree, this sequence is decomposed
as two branches because segments 0 (soma) and 9 (first segment in axon) are at the
root of the tree.

Similarly to segments, the branches in a morphology have a parent child relationship.
Every branch has one parent, with branches at the root of the tree having the placeholder
parent index ``mnpos``. Segments can have any non-negative number of children,
however by nature of their construction, no branch can have only one child: a branch has
either no children, or two or more children.
The parent-child information and segments for the morphology are summarised:

.. csv-table::
   :widths: 10, 10, 10, 10

   **Branch**, **Parent**, **Children**, **Segments**
   0,          ``mnpos``,  "[1, 2]",       "[0, 1, 2]"
   1,          0,          "[]",           "[3, 4]"
   2,          0,          "[3, 4]",       "[5]"
   3,          2,          "[]",           "[6]"
   4,          2,          "[]",           "[7, 8]"
   5,          ``mnpos``,  "[]",           "[9, 10]"

Gaps between segments do not influence branch creation, hence branches
can contain gaps between segments. Take the example of a morphology with
a gap between the soma and the axona and dendritic trees:

.. figure:: gen-images/detached_morph.svg
  :width: 800
  :align: center

The soma is part of branch 0, despite the gap:

.. csv-table::
   :widths: 10, 10, 10, 10

   **Branch**, **Parent**, **Children**, **Segments**
   0,          ``mnpos``,  "[1, 2]",       "[0, 1, 2]"
   1,          0,          "[]",           "[3, 4]"
   2,          0,          "[3, 4]",       "[5]"
   3,          2,          "[]",           "[6]"
   4,          2,          "[]",           "[7, 8]"
   5,          ``mnpos``,  "[]",           "[9]"

Tag information is not used when creating branches, so that a branch can
contain segments with different tags, which in our examples gives branches
that contain both soma and dendrite segments. For example, when building the
soma from multiple segments:

.. figure:: gen-images/stacked_morph.svg
  :width: 800
  :align: center

The morphology has the same number of branches as the other examples, with
multiple soma and dendrite segments in branch 0.

.. csv-table::
   :widths: 10, 10, 10, 10

   **Branch**, **Parent**, **Children**, **Segments**
   0,          ``mnpos``,  "[1, 2]",       "[0, 1, 2, 3, 4, 5]"
   1,          0,          "[]",           "[6, 7]"
   2,          0,          "[3, 4]",       "[8]"
   3,          2,          "[]",           "[9]"
   4,          2,          "[]",           "[10, 11]"
   5,          ``mnpos``,  "[]",           "[12, 13]"

.. Note::

    Users of NEURON who are used to creating a separate soma section
    that is always the first section in a morphology should not
    worry that the soma is not treated as a special branch
    in the examples above.

    Arbor provides a consistent representation of morphologies with no
    special cases for concepts like magical soma branches, so that we
    can build reproducable and consistent model descriptions.

    The soma in the examples above can be referred to in later model
    building phases, for example when describing the distribution of
    ion channels, by using refering to all parts of the cell with
    :ref:`tag 1 <labels-expressions>`.


Examples
~~~~~~~~~~~~~~~

Here we present a series of morphology examples of increasing complexity.
The examples use the Python API, and to simplify illustration, are two-dimensional
with the z-dimension set to zero.

.. _morph-tree1:

Example 1: Spherical cell
""""""""""""""""""""""""""""""

A simple model of a cell as a sphere can be modeled using a cylinder with length
and diameter equal to the diameter of the sphere, which will have the same
surface area (disregarding the area of the cylinder's circular ends).

Here a cylinder of length and diameter 5 μm is used to represent a *spherical cell*
with a radius of 2 μm, centered at the origin.

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint(-2, 0, 0, 2), mpoint(2, 0, 0, 2), tag=1)
    morph = arbor.morphology(tree)

.. figure:: gen-images/sphere_morph.svg
  :width: 400
  :align: center

  The morphology is a single cylinder segment (left) that forms branch 0 (right).

.. _morph-tree2:

Example 2: Unbranched cable
""""""""""""""""""""""""""""""

Consider a cable of length 10 μm, with a radius that tapers from 1 μm to 0.5 μm
at the proximal and distal ends respectively.
This can be described using a single segment.

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint(0, 0, 0, 1), mpoint(10, 0, 0, 0.5), tag=3)
    morph = arbor.morphology(tree)

.. figure:: gen-images/branch_morph1.svg
  :width: 600
  :align: center

  A tapered cable with one cable segment (left), generates a morphology with one branch (right).

The radius of a cable segment varies lineary between its end points. To define an unbranched cable
with irregular radius and "squiggly" shape, use multiple segments to build a piecewise linear reconstruction
of the cable geometry.
This example starts and ends at the same locations as the previous, however it is constructed from 4
distinct cable segments:

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint( 0.0,  0.0,  0.0, 1.0), mpoint( 3.0,  0.2,  0.0, 0.8), tag=1)
    tree.append(0,     mpoint( 3.0,  0.2,  0.0, 0.8), mpoint( 5.0, -0.1,  0.0, 0.7), tag=2)
    tree.append(1,     mpoint( 5.0, -0.1,  0.0, 0.7), mpoint( 8.0,  0.0,  0.0, 0.6), tag=2)
    tree.append(2,     mpoint( 8.0,  0.0,  0.0, 0.6), mpoint(10.0,  0.0,  0.0, 0.5), tag=3)
    morph = arbor.morphology(tree)

.. figure:: gen-images/branch_morph2.svg
  :width: 600
  :align: center

  The morphology is an ubranched cable comprised of 4 cable segments,
  colored according to their tags: tag 1 red; tag 2 gree; tag 3 blue (left).
  The four segments form one branch (right).

*TODO: gap jump description*
Gaps are possible between two segments. The example below inserts a 1 μm gap between the second
and third segments of the previous morphology. Note that Arbor will ignore the gap, effectively
joining the segments together, such that the the morphology with the gap is the same as that without.

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint( 0.0,  0.0,  0.0, 1.0), mpoint(3.0,  0.2,  0.0, 0.8), tag=1)
    tree.append(0,     mpoint( 3.0,  0.2,  0.0, 0.8), mpoint(5.0, -0.1,  0.0, 0.7), tag=2)
    tree.append(1,     mpoint( 7.0, -0.1,  0.0, 0.7), mpoint(10.0, 0.0,  0.0, 0.6), tag=2)
    tree.append(2,     mpoint(10.0,  0.0,  0.0, 0.6), mpoint(12.0, 0.0,  0.0, 0.5), tag=3)
    morph = arbor.morphology(tree)

.. figure:: gen-images/branch_morph3.svg
  :width: 600
  :align: center

  There is a gap between segment 1 and segment 2 (left), and there is a single branch (right).

The radius of a cable is piecewise linear, with discontinuities permited at the
interface between segments.
The next example adds a discontinuity to the previous example between segments
3 and 4, where the radius changes from 0.5 μm to 0.3 μm:

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint( 0.0,  0.0,  0.0, 1.0), mpoint( 3.0,  0.2,  0.0, 0.8), tag=1)
    tree.append(0,     mpoint( 3.0,  0.2,  0.0, 0.8), mpoint( 5.0, -0.1,  0.0, 0.7), tag=2)
    tree.append(1,     mpoint( 5.0, -0.1,  0.0, 0.7), mpoint( 8.0,  0.0,  0.0, 0.5), tag=2)
    tree.append(2,     mpoint( 8.0,  0.0,  0.0, 0.3), mpoint(10.0,  0.0,  0.0, 0.5), tag=3)
    morph = arbor.morphology(tree)

.. figure:: gen-images/branch_morph4.svg
  :width: 600
  :align: center

  The resulting morphology has a step discontinuity in radius.

.. _morph-example4:

Example 3: Y-shaped cell
""""""""""""""""""""""""""""""

The simplest branching morphology is a cable that bifurcates into two branches,
which we will call a *y-shaped cell*.
In the example below, the first branch of the tree is a cable of length 10 μm with a
a radius that tapers from 1 μm to 0.5 μm.
The two child branches are attached to the end of the first branch, and taper from from 0.5 μ m
to 0.2 μm.

*TODO* code below needs updating. The example itself is solid.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 1.0, tag= 3)
   tree.append(parent= 0, x=10.0, y= 0.0, z= 0.0, radius= 0.5, tag= 3)
   tree.append(parent= 1, x=15.0, y= 3.0, z= 0.0, radius= 0.2, tag= 3)
   tree.append(parent= 1, x=15.0, y=-3.0, z= 0.0, radius= 0.2, tag= 3)

   morph = arbor.morphology(tree, spherical_root=False)

.. figure:: gen-images/morph3a.svg
  :width: 800
  :align: center


Example 4: Ball and stick
""""""""""""""""""""""""""""""

*TODO: (a) model soma as a cylinder attached to the stick (b) add a gap between ball and stick.*

The next example is a spherical soma of radius 2 μm with a single branch of length
8 μm and constant radius of 1 μm attached.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 2.0, tag= 1)
   tree.append(parent= 0, x= 2.0, y= 0.0, z= 0.0, radius= 1.0, tag= 1)
   tree.append(parent= 1, x=10.0, y= 0.0, z= 0.0, radius= 1.0, tag= 3)
   morph = arbor.morphology(tree, spherical_root=True)

   morph = arbor.morphology(tree, spherical_root=False)

.. figure:: gen-images/morph4a.svg
  :width: 600
  :align: center

*TODO: discuss spherical soma approximation with a cylinder here.*
To achieve the desired model of a spherical soma with a single cable segment attached,
generate the morphology with ``spherical_root=True``:

.. code:: Python

   morph = arbor.morphology(tree, spherical_root=True)

.. figure:: gen-images/morph4b.svg
  :width: 600
  :align: center

  The spherical root is a special branch with id 0, and the dendrite is a second branch numbered 1.

Example 5: Soma with y-shaped dendrites
"""""""""""""""""""""""""""""""""""""""

This example models a cell with a simple dendritic tree attached to a soma.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 3.0, tag= 1)
   tree.append(parent= 0, x= 5.0, y=-1.0, z= 0.0, radius= 0.8, tag= 1)
   tree.append(parent= 1, x=10.0, y= 0.5, z= 0.0, radius= 0.5, tag= 3)
   tree.append(parent= 2, x=15.0, y= 0.0, z= 0.0, radius= 0.5, tag= 3)
   tree.append(parent= 3, x=18.0, y= 5.0, z= 0.0, radius= 0.3, tag= 2)
   tree.append(parent= 4, x=23.0, y= 8.0, z= 0.0, radius= 0.3, tag= 2)
   tree.append(parent= 3, x=20.0, y=-4.0, z= 0.0, radius= 0.3, tag= 3)

   morph = arbor.morphology(tree)

The root sample with id 0 has a large radius to represent the soma, and the dendritic
tree is represented by samples 1-6.

the soma is treated as a truncated cone whose end points are defined by between samples 0 and 1.

.. figure:: gen-images/morph5a_cable.svg
  :width: 800
  :align: center

  **Left**: The entire cell is composed of frustums.
  **Right**: There are three branches, with branch 0 containing both the soma and the first dendrite.

If the first sample is treated as a spherical soma by setting ``spherical_root=True``, the
morphology has 4 branches, with the soma having its own spherical branch, and the dendritic tree
composed of 3 branches.

If the intention is to model a spherical soma with a cable segment attached to its surface,
an additional sample can be added at the edge of the soma to bridge the gap.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 3.0, tag= 1)
   tree.append(parent= 0, x= 3.0, y=-0.8, z= 0.0, radius= 0.8, tag= 1)
   tree.append(parent= 1, x= 5.0, y=-1.0, z= 0.0, radius= 0.8, tag= 3)
   tree.append(parent= 2, x=10.0, y= 0.5, z= 0.0, radius= 0.5, tag= 3)
   tree.append(parent= 3, x=15.0, y= 0.0, z= 0.0, radius= 0.5, tag= 3)
   tree.append(parent= 4, x=18.0, y= 5.0, z= 0.0, radius= 0.3, tag= 2)
   tree.append(parent= 5, x=23.0, y= 8.0, z= 0.0, radius= 0.3, tag= 2)
   tree.append(parent= 4, x=20.0, y=-4.0, z= 0.0, radius= 0.3, tag= 3)

.. figure:: gen-images/tree5b.svg
  :width: 400
  :align: center

  Sample tree with an additional sample added to the surface, at a distance of 3
  μm of the spherical root.

.. code:: Python

   morph = arbor.morphology(tree, spherical_root=True)

.. figure:: gen-images/morph5b_sphere.svg
  :width: 800
  :align: center

  The morphology has no gap between the soma and the start of the dendritic tree.

Example 6: Connecting branches to a soma
""""""""""""""""""""""""""""""""""""""""""

This example shows how to attach multiple branches to a soma when the location
where the branches are attached is important. Specifically, a cylindrical soma
with two dendrites branching from one end, and an axon hillock on the other end.
A cable segment is used to model the soma so that branches can be attached to
either it's proximal or distal end.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 2.0, tag= 1)
   tree.append(parent= 0, x= 6.0, y= 0.0, z= 0.0, radius= 2.0, tag= 1)
   tree.append(parent= 1, x= 6.0, y= 0.0, z= 0.0, radius= 0.5, tag= 3)
   tree.append(parent= 2, x=15.0, y= 5.0, z= 0.0, radius= 0.5, tag= 3)
   tree.append(parent= 3, x=20.0, y= 7.0, z= 0.0, radius= 0.3, tag= 3)
   tree.append(parent= 2, x=21.0, y=-3.0, z= 0.0, radius= 0.3, tag= 3)
   tree.append(parent= 0, x=-5.0, y= 0.0, z= 0.0, radius= 0.5, tag= 2)


.. figure:: gen-images/tree6.svg
  :width: 400
  :align: center

  Samples 0 and 1 have the same radius, and define the extent of the soma. Sample 2
  is collocated with sample 1, as the starting start of the two dendrites with radius
  0.5 μm. Sample 6 defines the narrow end of the axonal hillock, with the root as parent.

.. code:: Python

   morph = arbor.morphology(tree, spherical_root=False)

.. figure:: gen-images/morph6.svg
  :width: 800
  :align: center

  **Left**: The segmentation. The soma (red) is a cylinder, with the blue dendrite
  segments attached on its distal end. The axon hillock is modeled by the single
  grey tapered cable segment.

  **Right**: Branch 0 is the soma, the dendrites are branches 1 and 2, and the axon
  hillock is branch 3.

Python API
----------

.. currentmodule:: arbor

.. data:: mnpos
    :type: int

    Value used to indicate "no parent" in :class:`segment_tree` and :class:`morphology`
    trees of segments and branches respectively.

    `TODO: update this example to work with segment trees.`


    .. code-block:: python

        import arbor

        tree = arbor.sample_tree()

        # mnpos can be used to explicitly specify that the first sample
        # in the tree has no parent, though if the parent argument is not.
        # provided, it will default to mnpos.
        tree.append(parent=arbor.mnpos, x=0, y=0, z=10, radius=0.5, tag=1)
        tree.append(parent=0, x=0, y=10, z=10, radius=0.5, tag=1)
        tree.append(parent=0, x=0, y=20, z=10, radius=0.5, tag=1)

        # mnpos can also be used when querying a sample_tree or morphology,
        # for example the following snippet that finds all branches in the
        # morphology that are attached to the root of the morphology.
        m = arbor.morphology(tree)
        base_branches = [i for i in range(m.num_branches) if m.branch_parent(i) == arbor.mnpos]

        print(base_branches)


.. class:: location

    A location on :attr:`branch`, where :attr:`pos`, in the range ``0 ≤ pos ≤ 1``,
    gives the relative position
    between the proximal and distal ends of the branch. The position is in terms
    of branch length, so for example, on a branch of length 100 μm ``pos=0.2``
    corresponds to 20 μm and 80 μm from the proximal and distal ends of the
    branch respectively.

    .. function:: location(branch, pos)

        Constructor.

    .. attribute:: branch
        :type: int

        The branch id of the location.

    .. attribute:: pos
        :type: float

        The relative position of the location on the branch.

.. class:: cable

    An unbranched cable that is a subset of a branch.
    The values of ``0 ≤ prox ≤ dist ≤ 1`` are the relative position
    of the cable's end points on the branch, in terms
    of branch length. For example, on a branch of length 100 μm, the values
    :attr:`prox` =0.2, :attr:`dist` =0.8 describe a cable that starts and
    ends 20 μm and 80 μm along the branch respectively.

    .. function:: cable(branch, prox, dist)

        Constructor.

    .. attribute:: branch
        :type: int

        The branch id of the cable.

    .. attribute:: prox
        :type: float

        The relative position of the proximal end of the cable on the branch.

    .. attribute:: dist
        :type: float

        The relative position of the distal end of the cable on the branch.

.. class:: sample

    `TODO: replace with point description`

    `TODO: add segment description`

    A sample of a cell morphology at a fixed location in space. Describes the location
    of the sample as three-dimensional coordinates (:attr:`x`, :attr:`y`, :attr:`z`),
    the :attr:`radius` of the cable, and :attr:`tag` meta-data.

    .. attribute:: x
        :type: real

        X coordinate (μm)

    .. attribute:: y
        :type: real

        Y coordinate (μm)

    .. attribute:: z
        :type: real

        x coordinate (μm)

    .. attribute:: radius
        :type: real

        Radius of the cable (μm)

    .. attribute:: tag
        :type: int

        Integer tag meta-data associated with the sample.
        Typically the tag would correspond to the SWC structure identifier:
        soma=1, axon=2, dendrite=3, apical dendrite=4, however arbitrary
        tags, including zero and negative values, can be used.

.. class:: sample_tree

    `TODO: rewrite for segment_tree`

    A sample tree is a sample-based description of a cell's morphology
    Sample trees comprise a sequence of samples starting from a *root* sample,
    together with a parent-child adjacency relationship where a child sample is
    distal to its parent.
    Branches in the tree occur where a sample has more than one child.
    Furthermore, a sample can not have more than one parent.

    .. function:: sample_tree()

        Construct an empty sample tree.

    A morphology tree is constructed by *appending* samples to the tree.
    Samples are numbered starting at 0 in the order that they are added,
    with the first sample getting id 0, the second sample id 1, and so forth.

    A sample can not be added before its parent, hence the root of the sample
    tree is always the first to be added. In this manner, a sample tree is
    always guarenteed to be in a correct state, with consistent parent-child
    indexing, and with *n* samples numbered from *0* to *n-1*.

    If a *parent* index is not provided, the sample's parent is assumed to be
    the last sample added to the tree. Calls to append return the id that
    was given to the sample, which makes it.

    For example, to create a cell with a radius of 2 at the root and
    two branches attached to the root.

    .. code-block:: Python

        import arbor

        # Create an empty sample tree
        t = arbor.sample_tree()

        # Add a root sample with radius 2 (our soma)
        r = t.append(x=0, y=1, z=1, radius=2, tag=1)

        # Add a first sequence of 3 samples branching off the root:
        p = t.append(parent=r, x=2, y=0, z=0, radius=1, tag=1)
        p = t.append(parent=p, x=3, y=0, z=0, radius=1, tag=3)
        p = t.append(parent=p, x=4, y=0, z=0, radius=1, tag=3)

        # Create a second branch attached to the root with
        p = t.append(parent=r, x=0, y=2, z=0, radius=1, tag=1)
        p = t.append(parent=p, x=0, y=3, z=0, radius=1, tag=3)
        p = t.append(parent=p, x=0, y=4, z=0, radius=1, tag=3)

    Only the first sample in a contiguous sequence of samples that will
    form a branch need to explicitly provide a parent.
    In the following example we construct a y-shaped cell.

    .. code-block:: Python

        import arbor

        # Create an empty sample tree
        t = arbor.sample_tree()

        # Create a first branch with 3 samples.
        p = t.append(x=0, y=0, z=0, radius=1, tag=3)
        p = t.append(x=1, y=0, z=0, radius=1, tag=3)
        # Keep the index of the last sample in the branch to which
        # the child branches will be attached.
        b = t.append(x=2, y=0, z=0, radius=1, tag=3)

        # Create a second branch, branching off the first.
        p = t.append(parent=b, x=2, y=1, z=0, radius=1, tag=3)
        p = t.append(          x=2, y=2, z=0, radius=1, tag=3)
        p = t.append(          x=2, y=3, z=0, radius=1, tag=3)

        # Create a third branch, also attached to the first.
        # create the branch by explicitly using the last sample in the first
        # branch as the parent of the first sample in the branch.
        p = t.append(parent=b, x=2, y=0, z=1, radius=1, tag=3)
        p = t.append(          x=2, y=0, z=2, radius=1, tag=3)
        p = t.append(          x=2, y=0, z=3, radius=1, tag=3)

    .. function:: append(parent, x, y, z, radius, tag)

        Append a sample to the sample tree with parent sample with index ``parent``.

        :return: index of the new sample.

    .. function:: append(x, y, z, radius, tag)
        :noindex:

        Append a sample whose parent is the last sample added to the tree.

        :return: index of the new sample.

    .. function:: append(sample)
        :noindex:

        Append a sample whose parent is the last sample added to the tree.

        :return: index of the new sample.

    .. function:: append(parent, sample)
        :noindex:

        Append a sample to the sample tree with parent sample with index ``parent``.

        :return: index of the new sample.

    .. attribute:: empty
        :type: bool

        If the sample tree is empty (i.e. whether it has size 0)

    .. attribute:: size
        :type: int

        The number of samples.

    .. attribute:: parents
        :type: list

        The parent indexes.

    .. attribute:: samples
        :type: list

        The samples.

.. py:function:: load_swc(filename)

    `TODO: add description of "how" the swc samples are interpreted.`

    `TODO: Allen version.`

    Loads an SWC file as a :class:`segment_tree`.

    :rtype: segment_tree

.. py:class:: morphology

    A *morphology* describes the geometry of a cell as unbranched cables with variable radius,
    an optional spherical segment at the root of the tree, and their associated tree structure.

    .. note::
        A morphology takes a segment tree and construct the cable branches.
        Meta data about branches and their properties that may be expensive to calculate
        is stored for fast look up during later stages of model building, and
        querying by users.

        For this reason, morpholgies are read only. To change a morphology, a new
        morphology should be created using a new segment tree.

    There are two *constructors* for a morphology:

    .. function:: morphology(segment_tree)

        Construct from a segment tree.

    The morphology provides an interface for querying morphology properties:

    .. attribute:: empty
            :type: bool

            Indicates if the morphology is empty.

    .. attribute:: num_branches
            :type: int

            The number of branches in the morphology.

    .. function:: branch_parent(i)

            The parent branch of branch ``i``

            :rtype: int

    .. function:: branch_children(i)

            The child branches of branch ``i``

            :rtype: list

    .. function:: branch_segments(i)

            A list of the segments in branch ``i``, ordered from proximal to distal.

            :rtype: list

