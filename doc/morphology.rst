.. _morphology:

Morphology
==========

A cell's *morphology* describes both its geometry and branching structure.
Morphologies in Arbor are modelled as a set of one dimensional cables of variable radius,
joined together to form a tree. The geometry is given by a series of segments,
which are truncated conic frustums that represent an unbranched section of the cell.

Segment Trees
--------------

A *segment tree* is a segment-based description of a cell's morphology
that is designed to support both the diverse descriptions
of cell morphologies (e.g. SWC, NeuroLicida, NeuroML), and tools that
iteratively construct cell morphologies (e.g. L-system generators, interactive cell-builders).

The building block of segment trees is a *segment*, which
is a three-dimensionsal *location*, with a *radius* and *tag* meta-data.

.. csv-table:: Fields that define a point.
   :widths: 10, 10, 30

   **Field**,   **Type**, **Description**
   ``x``,       real, x coordinate of centre of cable.
   ``y``,       real, y coordinate of centre of cable.
   ``z``,       real, z coordinate of centre of cable.
   ``radius``,  real, cross sectional radius of cable.


.. note::

    A *tag* is an integer label on every segment, which can be used later to define
    regions on cell models. For example, tags could store the *structure identifier* field in the
    `SWC format <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_,
    which identifies whether individual SWC records lie in the soma, axons, dendrites, etc.
    Arbor tag definitions are not fixed, and users can customise them for their requirements.


Segment trees comprise a sequence of segments starting from a *root* segment, together with a parent-child
adjacency relationship where a child segment is distal to its parent.
Branches in the tree occur where a segment has more than one child.
Furthermore, a segment can not have more than one parent.
In this manner, neuron morphologies are modelled as a *tree*, where cables that represent dendrites and
axons can branch, but branches can not rejoin.

.. _morph-sample-definitions:

Definitions
~~~~~~~~~~~

When refering to segments in a segment tree, the following definitions are used:

* *root*: segments at the root or start of the tree. A non-empty tree must have at least one root segment,
  and the first segment will always be a root.

* *parent*: Each segment has one parent, which is either 

  * A segments's id is always greater than the id of its parent.
  * The ids of segments on the same unbranched sequence of segments do not need to be contiguous.

* *child*: The children of segment *s* are segments whose parent is *s*.
* *terminal*: A segment with no children. Terminals lie at the end of dendritic trees or axons.
* *fork*: A segment with more than one child. The distal end of a for segment are fork points,
  where a cable splits into two or more branches.

  * Arbor allows more than two branches at a fork point.

Some of these definitions are illustrated in the following segment tree:

.. _morph-stree-fig:

.. figure:: gen-images/stree.svg
  :width: 400
  :align: center
  :alt: A sample tree with 7 points, with root, fork and terminal samples marked.

  *TODO* replace with two segment trees, that demonstrate with and without gaps.

  .. csv-table::
       :widths: 10, 10, 10, 10, 10, 10, 10

       **id**,   **parent**, **x**, **y**, **z**, **radius**, **tag**
       0, .,  0.0,  0.0, 0.0, 3.0, 1
       1, 0,  5.0, -1.0, 0.0, 1.2, 1
       2, 1, 10.0,  0.5, 0.0, 1.2, 3
       3, 2, 15.0,  0.0, 0.0, 1.0, 3
       4, 3, 18.0,  5.0, 0.0, 1.0, 2
       5, 4, 23.0,  8.0, 0.0, 0.7, 2
       6, 3, 20.0, -4.0, 0.0, 0.8, 3

  The tag meta-data is described in more detail :ref:`below <morph-tags>`.


* The tree is composed of 7 samples, enumerated with ids from 0 to 6.
* The root of the tree is sample 0.
* Sample 3 is a fork point whose children are samples 4 and 6.
* Samples 5 and 6 are terminals, with no children.
* Every sample has one parent, except for the root sample.

.. _morph-morphology:

Morphology
----------

A *morphology* describes the geometry of a cell as unbranched cables with variable radius
, and their associated tree structure.

A morphology is constructed from a segment tree by defining the branches, which are uniquely
derived from the cable segments.

Segmentation generates *cable segments* from a sample tree, of which there are two kinds:

* *Cable segment*: a frustum (cylinder or truncated cone) between two adjacent samples,
  with the centre and radius of each end defined by the location and radius of the samples.

The segmentation below, based on the model of a soma with a branching dendrite :ref:`above <morph-stree-fig>`,
illustrates the segments generated from a sample tree:

.. _morph-segment-fig:

.. figure:: gen-images/morph-segments.svg
  :width: 800
  :align: center

  **Left**: stuff

  **Right**: Segments with a spherical root segment.


.. note::
    The surface of the spherical root segment above does not coincide with
    the first sample of the dendritic tree, forming a gap between the
    sphere and the start of the dendrite.
    Segments attached to a spherical root branch are modeled as though they
    were attached to a single location on the sphere's surface, regardless of where they
    start in space.

    A gap between a spherical root and segments attached to it does not mean
    that the segmentation is not valid.
    To illustrate why this can occur, consider a potato-shaped soma modeled with a
    sphere of the same surface area, where sample 1 is the location where the dendrite attaches
    to the potato soma.
    The cell model will correctly represent the location and dimension of the dendritic tree,
    while preserving the soma surface area with a simplified spherical model.

.. warning::

    *TODO: update with an observation about how spheres are not appropriate in cable models, full stop.*
    Spheres are not suitable for representing the soma when it is important to model the location
    of cables attached to the soma. For example, when differentiating between apical and distal
    dendrites, or the location of the axon relative to that of the dendritic tree.
    In these cases, construct the soma from one or more frustums, and attach the cables to
    the appropriate end of the frustums.

.. _morph-tags:

Tags
""""

*TODO: move to segment tree*
Each segment is given a tag, determined by sample tag meta-data:

* Cable segments take the tag of their distal sample.
* Spherical segments take the tag of the root sample.

The segments :ref:`above <morph-segment-fig>` are colored according to the tags in
the :ref:`sample tree  <morph-stree-fig>`: tag 1 pink; tag 2 grey; and tag 3 blue.

.. note::

    The tag of the root sample is ignored when not using a spherical root,
    because it can only be used as the proximal end of cable segments.


Branches
~~~~~~~~

The second step in constructing a morphology is to group
the segments that define the geometry of the cell into non-overlapping sets called *branches*,
of which there are two types:

* *cable branches*: unbranched sequences of cable segments that have one of root, fork,
  or terminal samples at the end, and no fork samples between.

  * At least one segment, and hence two samples that define its ends, are
    required to define a cable branch.

* *spherical branches*: branches composed of a single spherical segment.

Because the end points of a branch mush be root, fork or terminal, it is
not possible to subdivide a cable branch into two smaller branches.
As a result, the set of branches that describe a morphology is unique.

.. figure:: gen-images/morph-branches.svg
  :width: 800
  :align: center

  The branches from the two segmentations of the :ref:`example morphology <morph-segment-fig>`.

  **Left**: Branches generated without a spherical root segment. The segment at the root is
  part of the first dendrite cable branch:

   .. csv-table::
       :widths: 10, 10

       **Branch**,   **Samples**
       0,            "[0, 1, 2, 3]"
       1,            "[3, 4, 5]"
       2,            "[3, 6]"

  **Right**: An additional branch is created for a spherical root segment:

   .. csv-table::
       :widths: 10, 10

       **Branch**,   **Samples**
       0,            "[0]"
       1,            "[1, 2, 3]"
       2,            "[3, 4, 5]"
       3,            "[3, 6]"


Branches are numbered starting from 0, and are sorted according to the lowest
sample id in each branch.
If two branches have the same parent sample, which will always be the
lowest sample id on each branch, then the next lowest id in each branch
is used for ordering.

Take, for example, the left decomposition above with three branches.
The main dendrite is numbered 0 by virtue of containing sample 0.
Of the two child branches, the top branch is numbered first because while the
the lowest sample id in both is 3, the second lowest id of the two is sample 4
in the top branch.

Examples
~~~~~~~~~~~~~~~

Here we present a series of morphology examples of increasing complexity.
The examples use the Python API, and to simplify illustration, are two-dimensional
with the z-dimension set to zero.

.. _morph-tree1:

Example 1: Spherical cell
""""""""""""""""""""""""""""""

Here a single sample is used to represent a *spherical cell*
with a radius of 2 μm, centered at the origin.

.. code:: Python

    tree = arbor.sample_tree()
    tree.append(x= 0.0, y= 0.0, z= 0.0, radius=2.0, tag= 1)
    morph = arbor.morphology(tree)

.. figure:: gen-images/morph1.svg
  :width: 100
  :align: center

  The morphology is a single spherical segment that forms branch 0.

.. _morph-tree2:

Example 2: Unbranched cable
""""""""""""""""""""""""""""""

Consider a cable of length 10 μm, with a radius that tapers from 1 μm to 0.5 μm
at the proximal and distal ends respectively.
It is constructed from a sample tree of two points that define the end points of the cable.

.. code:: Python

    tree = arbor.sample_tree()
    tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 1.0, tag= 3)
    tree.append(parent= 0, x=10.0, y= 0.0, z= 0.0, radius= 0.5, tag= 3)
    morph = arbor.morphology(tree)

.. figure:: gen-images/morph2a.svg
  :width: 600
  :align: center

  **Left**: The morphology is a tapered cable with one cable segment. **Right**: The morphology has one branch, numbered 0.

The radius of a cable segment varies lineary between its end points. To define an unbranched cable
with irregular radius and "squiggly" shape, use multiple samples to build a piecewise linear reconstruction
of the cable geometry.
This example starts and ends at the same locations as the previous, however it is constructed from 4
distinct cable segments:

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 1.0, tag= 1)
   tree.append(parent= 0, x= 3.0, y= 0.2, z= 0.0, radius= 0.8, tag= 1)
   tree.append(parent= 1, x= 5.0, y=-0.1, z= 0.0, radius= 0.7, tag= 2)
   tree.append(parent= 2, x= 8.0, y= 0.0, z= 0.0, radius= 0.6, tag= 2)
   tree.append(parent= 3, x=10.0, y= 0.0, z= 0.0, radius= 0.5, tag= 3)

   morph = arbor.morphology(tree)

.. figure:: gen-images/morph2b.svg
  :width: 600
  :align: center

  **Left**: The morphology is an ubranched cable comprised of 4 cable segments.
  The color of the segments is determined by the tags of their distal samples:
  tag 1 red; tag 2 gree; tag 3 blue.

  **Right**: The four segments form one branch.

Collocated samples can be used to create a discontinuity in cable radius.
The next example adds a discontinuity to the previous example at sample 3, where the
radius changes from 0.5 μm to 0.3 μm:

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 1.0, tag= 1)
   tree.append(parent= 0, x= 3.0, y= 0.2, z= 0.0, radius= 0.8, tag= 1)
   tree.append(parent= 1, x= 5.0, y=-0.1, z= 0.0, radius= 0.7, tag= 2)
   tree.append(parent= 2, x= 8.0, y= 0.0, z= 0.0, radius= 0.6, tag= 2)
   tree.append(parent= 3, x= 8.0, y= 0.0, z= 0.0, radius= 0.3, tag= 3)
   tree.append(parent= 4, x=10.0, y= 0.0, z= 0.0, radius= 0.5, tag= 3)

   morph = arbor.morphology(tree)

.. figure:: gen-images/morph2c.svg
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

    Value used to indicate "no parent" in :class:`sample_tree` and :class:`morphology`
    trees of samples and branches respectively.


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
    of the ends of the branch. The positions are in terms
    of branch length, so for example, on a branch of length 100 μm where
    :attr:`prox` =0.2, :attr:`dist` =0.8 would give a cable that starts and
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

    Loads an SWC file as a :class:`sample_tree`.

    :rtype: sample_tree

.. py:class:: morphology

    A *morphology* describes the geometry of a cell as unbranched cables with variable radius,
    an optional spherical segment at the root of the tree, and their associated tree structure.

    .. note::
        A morphology takes a sample tree and a flag that indicates whether the
        morphology has a spherical root, to construct the cable segments and branches,
        which are unique for any sample tree.
        Meta data about branches and their properties that may be expensive to calculate
        is stored for fast look up during later stages of model building, and
        querying by users.

        For this reason, morpholgies are read only. To change a morphology, a new
        morphology should be created using a new sample tree.

    There are two *constructors* for a morphology:

    .. function:: morphology(sample_tree, spherical_root)

        Construct from a sample tree where the ``spherical_root`` argument
        is a bool that indicates whether the root sample is to be used as
        the centre and radius of a spherical segment/branch.

    .. function:: morphology(sample_tree)
        :noindex:

        Construct a morphology from a sample tree where it is
        infered whether the root is spherical or not: if the root
        sample has a different tag to all of its children the
        root is assumed to be spherical.

    The morphology provides an interface for querying morphology properties:

    .. attribute:: empty
            :type: bool

            Indicates if the morphology is empty.

    .. attribute:: spherical_root
            :type: bool

            Indicates if the root of the morphology is spherical.

    .. attribute:: num_branches
            :type: int

            The number of branches in the morphology.

    .. attribute:: num_samples
            :type: int

            The number of samples in the morphology.

    .. attribute:: samples
            :type: list

            The samples in the morphology.

    .. attribute:: sample_parents
            :type: list

            The parent indexes of each sample.

    .. function:: branch_parent(i)

            The parent branch of branch ``i``

            :rtype: int

    .. function:: branch_children(i)

            The child branches of branch ``i``

            :rtype: list

    .. function:: branch_indexes(i)

            Range of indexes into the sample points in branch ``i``.

            :rtype: list

