.. _morphology:

Morphology
==========

A cell's *morphology* describes both its geometry and branching structure.
Morphologies in Arbor are modeled as a set of one dimensional cables of variable radius,
joined together to form a tree.

The building blocks of morpholgies tree are points and segments.
A *point* is a three-dimensional location and a radius, used to mark the centre and radius
of the cable.

.. csv-table::
   :widths: 10, 10, 30

   **Field**,   **Type**, **Description**
   ``x``,       real, x coordinate of centre of cable (μm).
   ``y``,       real, y coordinate of centre of cable (μm).
   ``z``,       real, z coordinate of centre of cable (μm).
   ``radius``,  real, cross sectional radius of cable (μm).


A *segment* is a frustum (cylinder or truncated cone), with the centre and radius at each
end defined by a pair of points.

.. csv-table::
   :widths: 10, 10, 30

   **Field**,   **Type**, **Description**
   ``prox``,       :py:class:`point <arbor.mpoint>`,   the center and radius of the proximal end.
   ``dist``,       :py:class:`point <arbor.mpoint>`,   the center and radius of the distal end.
   ``tag``,        integer, tag meta-data.

.. _morph-tag-definition:

A *tag* is an integer label on every segment, which can be used to define disjoint
regions on cells.
The meaning of tag values are not fixed in Arbor, however we typically use tag values that correspond
to SWC `structure identifiers <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_.

.. _morph-segment-definitions:

Segment Trees
--------------

A *segment tree* describes a morphology as a set of segments and their connections,
designed to support both the diverse descriptions
of cell morphologies (e.g. SWC, NeuroLicida, NeuroML), and tools that
iteratively construct cell morphologies (e.g. L-system generators, interactive cell-builders).

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

* *parent*: Each segment has one parent, except for root segments which have :data:`mnpos <arbor.mnpos>` as their parent.

  * The id of a segment is always greater than the id of its parent.
  * The ids of segments on the same unbranched sequence of segments do not need to be contiguous.

* *child*: A segment's children are the segments that have the segment as their parent.
* *terminal*: A segment with no children. Terminals lie at the end of dendritic trees or axons.
* *fork*: A segment with more than one child. The distal end of a fork segment are *fork points*,
  where a cable splits into two or more branches.

  * Arbor allows more than two branches at a fork point.

The following segment tree models a soma as a cylinder, a branching dendritic tree and
an axon with an axonal hillock. The segments are colored according to their tag, which
in this case are SWC structure identifiers: tag 1 colored pink for soma;
tag 2 colored grey for axon; tag 3 colored blue for basal dendrites.

.. _morph-label-seg-fig:

.. figure:: gen-images/label_seg.svg
  :width: 600
  :align: center

  Example Python code to generate this morphology is in the :class:`segment_tree<arbor.segment_tree>`
  documentation :ref:`below <morph-label-seg-code>`.

We can apply the following labels to the segments:

* The tree is composed of 11 segments (1 soma, 2 axon, 8 dendrite).
* The proximal ends of segments 0 and 9 (the soma and axon hillock respectively) are attached to the root of the tree.
* Segment 2 is a fork, with segments 3 and 5 as children.
* Segment 5 is a fork, with segments 6 and 7 as children.
* There is also a fork at the root, whith segments 0 and 9 as children.
* Segments 4, 6, 8 and 10 are terminal segments.

In the example above there are no gaps between segments, however
it is possible for segments to be detached, where the proximal end of a segment is not coincident
with the distal end of its parent. The following morphology has gaps between the start of the
axon and dendritic tree and the soma segment to which they attach.

.. _morph-detached-seg-fig:

.. figure:: gen-images/detached_seg.svg
  :width: 600
  :align: center

.. note::
    In Arbor, segments are always treated as though they are connected directly
    to their parents, regardless of whether ends where they attached are collocated.

    Gaps are frequently the result of simplifying the soma,
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
, and their associated tree structure. 
Every segment tree can be used to generate a unique morphology, which derives and enumerates
*branches* from the segments.
The branches of a morphology are unbranched cables, composed of one or more segments, where:

  * the first (proximal) segment of the branch is either a root or the child of fork segment;
  * the last (distal) segment of the branch is either a fork or terminal segment;
  * branches are enumerated according to the ids of their proximal segments in the segment trree.

When constructed in this manner, the following statements are true for the branches and
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
which is illustrated along with its branches below.

.. _morph-label-morph-fig:

.. figure:: gen-images/label_morph.svg
  :width: 800
  :align: center

  The code used to generate this morphology is in the :class:`segment_tree<arbor.segment_tree>`
  documentation :ref:`below <morph-label-seg-code>`.

The first branch contains the soma and the first two segments of the dendritic tree.
There are four more branches in the dendritic tree, and one representing the two
segments of the axon.

Note, that though it is possible to create an unbranched sequence of segments composed
of the axon, soma and first two segements in the dendritic tree, this sequence is decomposed
as two branches because segments 0 (soma) and 9 (first segment in axon) are at the
root of the tree.

Similarly to segments, the branches in a morphology have a parent child relationship.
Every branch has one parent, with branches at the root of the tree having the placeholder
parent index :data:`mnpos <arbor.mnpos>`. Segments can have any non-negative number of children,
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
    Arbor provides a consistent representation of morphologies with no
    special cases for concepts like magical soma branches, in order to
    build reproducable and consistent model descriptions.

    Users of NEURON who are used to creating a separate soma section
    that is always the first section in a morphology should not
    worry that the soma is not treated as a special branch
    in the examples above.

    The soma in the examples above can be referred to in later model
    building phases, for example when describing the distribution of
    ion channels, by using refering to all parts of the cell with
    :ref:`tag 1 <labels-expressions>`.


Examples
~~~~~~~~~~~~~~~

Here we present a series of morphology examples of increasing complexity.
The examples use the Python API are two-dimensional, with the z-dimension set to zero.

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

    morph = arbor.morphology(tree)

.. figure:: gen-images/branch_morph2.svg
  :width: 600
  :align: center

  The morphology is an ubranched cable comprised of 4 cable segments,
  colored according to their tags: tag 1 red; tag 2 gree; tag 3 blue (left).
  The four segments form one branch (right).

Gaps are possible between two segments. The example below inserts a 1 μm gap between the second
and third segments of the previous morphology. Note that Arbor will ignore the gap, effectively
joining the segments together, such that the morphology with the gap is the same as that without.

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

Note that only the distal point is required to describe the child segments,
because the proximal end of each child segment has the same location and
radius as the distal end of the parent.

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint( 0.0, 0.0, 0.0, 1.0), mpoint(10.0, 0.0, 0.0, 0.5), tag= 3)
    tree.append(0,     mpoint(15.0, 3.0, 0.0, 0.2), tag= 3)
    tree.append(0,     mpoint(15.0,-3.0, 0.0, 0.2), tag= 3)
    morph = arbor.morphology(tree)

.. figure:: gen-images/yshaped_morph.svg
  :width: 800
  :align: center

Example 4: Soma with branches
""""""""""""""""""""""""""""""

Now let's look at cell with a simple dendritic tree attached to a spherical soma.
The spherical soma of radius 3 μm is modelled with a cylinder with length and
diameter equal to 6 μm, which has the same surface area as the sphere.

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint(-3.0, 0.0, 0.0, 3.0), mpoint( 3.0, 0.0, 0.0, 3.0), tag=1)
    tree.append(0, mpoint( 4.0, -1.0,  0.0, 0.6), mpoint(10.0,  -2.0,  0.0, 0.5), tag=3)
    tree.append(1, mpoint(15.0, -1.0,  0.0, 0.5), tag=3)
    tree.append(2, mpoint(18.0, -5.0,  0.0, 0.3), tag=3)
    tree.append(2, mpoint(20.0,  2.0,  0.0, 0.3), tag=3)
    morph = arbor.morphology(tree)


.. figure:: gen-images/ysoma_morph1.svg
  :width: 900
  :align: center

  Note that branch 0 (right) is composed of segments 0, 1, and 2 (left).

The soma is the first segment, labeled with tag 1. The dendritic tree is a simple
y-shaped tree composed of 4 segments, each labeled with tag 3.
The first branch is composed of 3 segments: the soma segment and the first two segments
in the dendritic tree because the segments have parent child ordering and no fork points.

.. note::
    The first branch is derived directly from the topological relationship between the segments,
    and no special treatment is given to the soma.
    There is no need to treat segments with different tags (e.g. tags that we might associate
    with soma, axon, basal dendrite and apical dendrite) when defining geometric primitives like
    segments and branches, because they can later be referenced later using
    :ref:`region expressions <labels-expressions>`.

Now we can attach another dendrite and an axon to the soma, to make a total of three cables
attached to the soma (two dendrites and an axon).
The dendrites are attached to the distal end of the soma (segment 0), so they have the
0 as their parent.
The axon is attached to the proximal end of the soma, which is at the root of the tree,
so it has :data:`mnpos` as its parent.
There are 7 branches generated from 10 segments, and soma segment is its own branch,
because it has two children: the dendrites attached to its distal end.

.. figure:: gen-images/ysoma_morph2.svg
  :width: 900
  :align: center


.. note::
    The discretization process, which converts segments and branches into compartments,
    will ignore gaps between segments in the input. The cell below, in which the dendrites
    and axon have been translated to remove any gaps, is equivalent to the previous example
    for the back end simulator.

    Note that the dendrites are children of the soma segment, so they are coincident with
    the distal end of the soma, and the axon is translated to the proximal end of the
    soma segment because both it and the soma have :py:data:`mnpos <arbor.mnpos>` as a parent.
    More generally, segments at the root of the tree are connected electrically at their
    proximal ends.

    .. figure:: gen-images/ysoma_morph3.svg
      :width: 900
      :align: center

Python API
----------

.. currentmodule:: arbor

.. data:: mnpos
    :type: int

    Value used to indicate "no parent" in :class:`segment_tree` and :class:`morphology`
    trees of segments and branches respectively.

    .. code-block:: python

        import arbor

        tree = arbor.segment_tree()

        # mnpos can be used to explicitly specify that a segment
        # is at the root of the tree. More than one segment can
        # be at the root, and they will all be joined electrically
        # at their proximal ends.
        tree.append(parent=arbor.mnpos, # attach segment to root.
                    prox=arbor.mpoint(0, 0,-5, 5),
                    dist=arbor.mpoint(0, 0, 5, 5),
                    tag=1)
        tree.append(parent=0,
                    prox=arbor.mpoint(0, 0, 5, 0.5),
                    dist=arbor.mpoint(0, 0,50, 0.2),
                    tag=3)

        # mnpos can also be used when querying a sample_tree or morphology,
        # for example the following snippet that finds all branches in the
        # morphology that are attached to the root of the morphology.
        m = arbor.morphology(tree)
        base_branches = [i for i in range(m.num_branches)
                            if m.branch_parent(i) == arbor.mnpos]

        print(base_branches)



.. class:: location

    A location on :attr:`branch`, where :attr:`pos`, in the range ``0 ≤ pos ≤ 1``,
    gives the relative position
    between the proximal and distal ends of the branch. The position is in terms
    of branch path length, so for example, on a branch of path length 100 μm ``pos=0.2``
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
    of branch path length. For example, on a branch of path length 100 μm, the values
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

.. class:: mpoint

    A location of a cell morphology at a fixed location in space. Describes the location
    of the as three-dimensional coordinates (:attr:`x`, :attr:`y`, :attr:`z`) and
    the :attr:`radius` of the cable.

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

.. class:: segment

    .. attribute:: prox
        :type: mpoint

        The location and radius at the proximal end of the segment.

    .. attribute:: dist
        :type: mpoint

        The location and radius at the distal end of the segment.

    .. attribute:: tag
        :type: int

        Integer tag meta-data associated with the segment.
        Typically the tag would correspond to the SWC structure identifier:
        soma=1, axon=2, dendrite=3, apical dendrite=4, however arbitrary
        tags, including zero and negative values, can be used.

.. class:: segment_tree

    A segment tree is a description of a the segments and their connections
    Segment trees comprise a sequence of segments starting from at lease one *root* segment,
    together with a parent-child adjacency relationship where a child segment is
    distal to its parent.
    Branches in the tree occur where a segment has more than one child.
    Furthermore, a segment can not have more than one parent.
    In this manner, neuron morphologies are modeled as a *tree*, where cables that
    represent dendrites and axons can branch, but branches can not rejoin.
    A segment tree is a segment-based description of a cell's morphology.

    .. function:: segment_tree()

        Construct an empty segment tree.

    The tree is constructed by *appending* segments to the tree.
    Segments are numbered starting at 0 in the order that they are added,
    with the first segment getting id 0, the second segment id 1, and so forth.

    A segment can not be added before its parent, hence the first segment
    is always at the root. In this manner, a segment tree is
    always guarenteed to be in a correct state, with consistent parent-child
    indexing, and with *n* segments numbered from *0* to *n-1*.

    To illustrate how a segment tree is constructed by appending segments,
    take the segment tree used in the :ref:`documentation above <morph-label-seg-fig>`.


    .. figure:: gen-images/label_seg.svg


    Which is constructed as follows.

    .. _morph-label-seg-code:

    .. code-block:: Python

        import arbor
        from arbor import mpoint
        from arbor import mpos

        tree = arbor.segment_tree()
        # Start with a cylinder segment for the soma (with tag 1)
        tree.append(mnpos, mpoint(0,   0.0, 0, 2.0), mpoint( 4,  0.0, 0, 2.0), tag=1)
        # Construct the first section of the dendritic tree,
        # comprised of segments 1 and 2, attached to soma segment 0.
        tree.append(0,     mpoint(4,   0.0, 0, 0.8), mpoint( 8,  0.0, 0, 0.8), tag=3)
        tree.append(1,     mpoint(8,   0.0, 0, 0.8), mpoint(12, -0.5, 0, 0.8), tag=3)
        # Construct the rest of the dendritic tree.
        tree.append(2,     mpoint(12, -0.5, 0, 0.8), mpoint(20,  4.0, 0, 0.4), tag=3)
        tree.append(3,     mpoint(20,  4.0, 0, 0.4), mpoint(26,  6.0, 0, 0.2), tag=3)
        tree.append(2,     mpoint(12, -0.5, 0, 0.5), mpoint(19, -3.0, 0, 0.5), tag=3)
        tree.append(5,     mpoint(19, -3.0, 0, 0.5), mpoint(24, -7.0, 0, 0.2), tag=3)
        tree.append(5,     mpoint(19, -3.0, 0, 0.5), mpoint(23, -1.0, 0, 0.2), tag=3)
        tree.append(7,     mpoint(23, -1.0, 0, 0.2), mpoint(26, -2.0, 0, 0.2), tag=3)
        # Two segments that define the axon, with the first at the root, where its proximal
        # end will be connected with the proximal end of the soma segment.
        tree.append(mnpos, mpoint(0,   0.0, 0, 2.0), mpoint(-7,  0.0, 0, 0.4), tag=2)
        tree.append(9,     mpoint(-7,  0.0, 0, 0.4), mpoint(-10, 0.0, 0, 0.4), tag=2)

    .. method:: append(parent, prox, dist, tag)

        Append a segment to the tree.

        :return: index of the new segment
        :param int parent: index of segment
        :param mpoint prox: proximal end of the segment
        :param mpoint dist: distal end of the segment
        :param int tag: tag meta data of segment

    .. method:: append(parent, dist, tag)
        :noindex:

        Append a segment to the tree whose proximal end has the location and
        radius of the distal end of the parent segment.

        This version of append can't be used for a segment at the root of the
        tree, that is, when ``parent`` is :data:`mnpos`, in which case both proximal
        and distal ends of the segment must be specified.

        :return: index of the new segment
        :param int parent: index of segment
        :param mpoint dist: distal end of the segment
        :param int tag: tag meta data of segment

    .. method:: append(parent, x, y, z, radius, tag)
        :noindex:

        Append a segment to the tree whose proximal end has the location and
        radius of the distal end of the parent segment.

        This version of append can't be used for a segment at the root of the
        tree, that is, when ``parent`` is :data:`mnpos`, in which case both proximal
        and distal ends of the segment must be specified.

        :return: index of the new segment
        :param int parent: index of segment
        :param float x: distal x coordinate (μm)
        :param float y: distal y coordinate (μm)
        :param float z: distal z coordinate (μm)
        :param float radius: distal radius (μm)
        :param int tag: tag meta data of segment

    .. attribute:: empty
        :type: bool

        If the tree is empty (i.e. whether it has size 0)

    .. attribute:: size
        :type: int

        The number of segments.

    .. attribute:: parents
        :type: list

        A list of parent indexes of the segments.

    .. attribute:: segments
        :type: list

        A list of the segments.

.. py:function:: load_swc(filename)

    Loads the morphology in an SWC file as a :class:`segment_tree`.

    The samples in the SWC files are treated as the end points of segments, where a
    sample and its parent form a segment.
    The :py:attr:`tag <segment.tag>` of each segment is the
    `structure identifier <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_
    of the distal sample.
    The structure identifier of the first (root) sample is ignored, as it can only be the
    proximal end of any segment.

    .. note::
        This method does not interpret the first sample, typically associated with the soma,
        as a sphere. SWCs with single point somas are, unfortunately, reasonably common, for example
        `SONATA <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#representing-biophysical-neuron-morphologies>`_
        model descriptions.

        Such representations are unfortunate because simulation tools like Arbor and NEURON require
        the use of cylinders or fustrums to describe morphologies, and it is not possible to
        infer how branches attached to the soma should be connected.

        The :func:`load_swc_allen` function provides support for interpreting
        such SWC files.


    :param str filename: the name of the SWC file.
    :rtype: segment_tree

.. py:function:: load_swc_allen(filename, no_gaps=False)

    Generate a segment tree from an SWC file following the rules prescribed by
    AllenDB and Sonata. Specifically:

        * The first sample (the root) is treated as the center of the soma.
        * The morphology is translated such that the soma is centered at (0,0,0).
        * The first sample has tag 1 (soma).
        * All other samples have tags 2, 3 or 4 (axon, apic and dend respectively)

    SONATA prescribes that there should be no gaps, however some models in AllenDB
    have gaps between the start of sections and the soma. The ``no_gaps`` flag can be
    used to enforce this requirement.

    Arbor does not support modelling the soma as a sphere, so a cylinder with length
    equal to the soma diameter is used. The cylinder is centered on the origin, and
    aligned along the z axis.
    Axons and apical dendrites are attached to the proximal end of the cylinder, and
    dendrites to the distal end, with a gap between the start of each branch and the
    end of the soma cylinder to which it is attached.

    :param str filename: the name of the SWC file.
    :param bool no_gaps: enforce that distance between soma center and branches attached to soma is the soma radius.
    :rtype: segment_tree

.. py:class:: morphology

    A *morphology* describes the geometry of a cell as unbranched cables
    with variable radius and their associated tree structure.

    .. note::
        A morphology takes a segment tree and construct the cable branches.
        Meta data about branches and their properties that may be expensive to calculate
        is stored for fast look up during later stages of model building, and
        querying by users.

        For this reason, morpholgies are read only. To change a morphology, a new
        morphology should be created using a new segment tree.

    There is one *constructor* for a morphology:

    .. function:: morphology(segment_tree)

        Construct from a segment tree.

    The morphology provides an interface for querying morphology properties:

    .. attribute:: empty
            :type: bool

            Indicates if the morphology is empty.

    .. attribute:: num_branches
            :type: int

            The number of branches in the morphology.

    .. method:: branch_parent(i)

            The parent branch of a branch.

            :param int i: branch index
            :rtype: int

    .. method:: branch_children(i)

            The child branches of a branch.

            :param int i: branch index
            :rtype: list

    .. method:: branch_segments(i)

            A list of the segments in a branch, ordered from proximal to distal.

            :param int i: branch index
            :rtype: list

