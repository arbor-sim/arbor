.. _morphology:

Morphology
==========

A cell's *morphology* describes both its geometry and branching structure. Morphologies in Arbor are modelled as a set of one dimensional cables of variable radius, joined together to form a tree. The geometry is given by a series of sample points, which comprise a three-dimensional location and a cross-sectional radius of the cell at that point.

Sample Trees
------------

A *sample tree* is a sample-based description of a cell's morphology
that is simple and flexible to support the diverse descriptions
of cell morphologies (e.g. SWC, NeuroLicida, NeuroML), and to support tools that
iteratively construct cell morphologies (e.g. L-system generators, interactive cell-builders).

The building block used to define a morphology in a sample tree is a *sample*, which
is a three-dimensionsal *location*, with a *radius* and *tag* meta-data.

.. csv-table:: Fields that define a *sample*.
   :widths: 10, 10, 30

   **Field**,   **Type**, **Description**
   ``x``,       real, x coordinate of centre of cable.
   ``y``,       real, y coordinate of centre of cable.
   ``z``,       real, z coordinate of centre of cable.
   ``radius``,  real, cross sectional radius of cable.
   ``tag``,     integer, label used to mark regions on the morphology.


.. note::

    A *tag* is an integer label on every sample, which can be used later to define
    regions on cell models. For example, tags could store the *structure identifier* field in the
    `SWC format <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_,
    which identifies whether samples lie in the soma, axons, dendrites, etc. In Arbor tag definitions
    are not fixed, and users can customise them for their requirements.


Sample trees comprise a sequence of samples starting from a *root* sample, together with a parent-child
adjacency relationship where a child sample is distal to its parent 
Branches in the tree occur where a parent sample has more than one child, and a sample can not have more than one parent.
In this manner, neuron morphologies are modelled as a *tree*, where cables that represent dendrites and axons can branch, but branches can not rejoin.

* *root*: The first sample in the sample tree, assigned id 0.

  * later xxx

* *parent*: Each sample has one parent parent, except the root sample.

  * A sample's id is always greater than the id of its parent.
  * The ids of samples on the same unbranched section of cable do not need to be contiguous.

* *child*: The children of sample *s* are samples whose parent is *s*.
* *terminal*: A sample with no children.
* *fork*: A sample with more than one child. Fork points are where a cable splits into two or more branches.
* *collocated*: A sample is collocated with its parent if they have the same location

  * Used in to indicate a discontinuity in the radius of a cable, or the start of a
    child branch with a different radius than its parent.

Some of these concepts are illustrated in the sample tree :ref:`below <morph-stree-fig>`:

* The tree is composed of 7 samples, enumerated from 0 to 6.
* The root of the tree is sample 0.
* Sample 3 is a fork point whose children are samples 4 and 6.
* Samples 5 and 6 are terminals, with no children.
* Every sample has one parent, except for the root sample.

.. _morph-stree-fig:

.. figure:: gen-images/stree.svg
  :width: 400
  :align: center
  :alt: A sample tree with 7 points, with root, fork and terminal samples marked.

  A sample tree with 7 samples defined as follows:

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

.. _morph-morphology:

Morphology
----------

A *morphology* provides an description of the geometry of a cell in terms of variable radius unbranched cables, an optional spherical segment at the root of the tree, and their associated tree structure.

Segmentation
~~~~~~~~~~~~

The first step in constructing a morphology from a sample tree is to generate *segments*, of which there are two kinds:

* *cable segment*: a frustum (cylinder or truncated cone) between two adjacent samples,
  with the centre and radius of each end defined by the location and radius of the samples.
* *spherical segment*: a sphere with centre and radius specified by the location and radius
  of the root sample. Only the root sample can be interpreted as a spherical segment,
  which is a user option.

The following example, based on the model of a soma with a branching dendrite above, illustrates the segments generated from a sample tree.

.. _morph-segment-fig:

.. figure:: gen-images/tree5a.svg
  :width: 400
  :align: center

  Sample tree with 7 samples.

.. figure:: gen-images/morph-segments.svg
  :width: 800
  :align: center

  **Left**: The segments generated without a spherical root.

  **Right**: Segments with a spherical root segment.


The surface of  the spherical root segment above does not conincide with
the first sample of the dendritic tree, so there is a gap between the
sphere and the start of the dendrite.
This does not neccesarily mean that the segmentation is not valid.

To illustrate why, consider a potato-shaped soma modeled with a sphere of the
same surface area, where sample 1 is the location where the dendrite attaches
to the potato soma.
Segments attached to a spherical root branch are modeled as though they
were attached to a single location on the sphere's surface, regardless of where they
start in space.

.. warning::

    Spheres are not suitable for representing the soma when it is important to model the location
    of cables attached to the soma. For example, differentiating between apical and distal
    dendrites, or the location of the axon hillock.
    In this case, construct the soma from one or more frustums, and attach the cables to
    the end points of the frustums.

.. _morpho-tags:

Tags
~~~~

The tag meta-data attached to each sample is used to attach tags to segments.

* Cable segments take the tag of the distal sample.
* Spherical segments get the tag of the root samle.

The segments :ref:`above <morph-segment-fig>` are colored according to the tags in
the :ref:`sample tree  <morph-stree-fig>`: tag 1 pink; tag 2 green; and tag 3 blue.

.. note::

    The tag of the root sample is ignored when not using a spherical root,
    because it is the proximal end of any cable segment.


Branches
~~~~~~~~

The morphology groups the segments that define the geometry of the cell into non-overlapping
sets called branches. There are two types of branch:

* *spherical branch*: branch composed of a single spherical segment.
* *cable branch*: an unbranched sequence of cable segments that has one of root, fork, or terminal samples at the end, and has no fork samples between.

  * At least one segment, and hence two samples to define its ends, are required to make a cable branch

Because the end points of a branch mush be root, fork or terminal, it is not possible to
subdivide a cable branch into two smaller branches.
As a result, there is only one possible set of branches that describe a morphology.

.. figure:: gen-images/morph-branches.svg
  :width: 800
  :align: center

  The branches from the segmentations without and with a spherical root.

  **Left**: The branches generated with no spherical root. The segment at the root is
  part of the first dendrite cable branch:

   .. csv-table::
       :widths: 10, 10

       **Branch**,   **Samples**
       0,            "[0, 1, 2, 3]"
       1,            "[3, 4, 5]"
       2,            "[3, 6]"

  **Right**: An additional branch is created for a spherical root segment, with only the root sample:

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
with a radius of 3 μm, centered at the origin.

.. code:: Python

    tree = arbor.sample_tree()
    tree.append(x= 0.0, y= 0.0, z= 0.0, radius=2.0, tag= 1)

.. figure:: gen-images/tree1.svg
  :width: 100
  :align: center

  The sample tree is a single sample with radius 3 μm.

.. code:: Python

    morph = arbor.morphology(tree, spherical_root=True)

.. figure:: gen-images/morph1.svg
  :width: 100
  :align: center

  The corresponding morphology is a sphere of radius 3 μm.

.. _morph-tree2:

Example 2: Unbranched cable
""""""""""""""""""""""""""""""

Consider a cable of length 10 μm, with a radius that tapers from 1 μm to 0.5 μm
at the proximal and distal ends respectively.
It is constructed from a sample tree of two points that define the end points of the cable.

.. code:: Python

    tree = arbor.sample_tree()
    tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 1.0, tag= 1)
    tree.append(parent= 0, x=10.0, y= 0.0, z= 0.0, radius= 0.5, tag= 1)

.. figure:: gen-images/tree2a.svg
  :width: 300
  :align: center

  The sample tree has a root and terminal point, marked in blue and green respectively.

.. code:: Python

    morph = arbor.morphology(tree, spherical_root=False)

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
   tree.append(parent= 1, x= 5.0, y=-0.1, z= 0.0, radius= 0.7, tag= 1)
   tree.append(parent= 2, x= 8.0, y= 0.0, z= 0.0, radius= 0.6, tag= 1)
   tree.append(parent= 3, x=10.0, y= 0.0, z= 0.0, radius= 0.5, tag= 1)

.. figure:: gen-images/tree2b.svg
  :width: 300
  :align: center

  The sample tree has 5 samples.

.. code:: Python

    morph = arbor.morphology(tree, spherical_root=False)

.. figure:: gen-images/morph2b.svg
  :width: 600
  :align: center

  **Left**: The resulting morphology is an ubranched cable comprised of 4 cable segments.
  **Right**: The four segments form one branch.

Collocated samples can be used to create a discontinuity in cable radius.
The next example adds a discontinuity to the previous example at sample 3, where the
radius changes from 0.5 μm to 0.3 μm:

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 1.0, tag= 1)
   tree.append(parent= 0, x= 3.0, y= 0.2, z= 0.0, radius= 0.8, tag= 1)
   tree.append(parent= 1, x= 5.0, y=-0.1, z= 0.0, radius= 0.7, tag= 1)
   tree.append(parent= 2, x= 8.0, y= 0.0, z= 0.0, radius= 0.6, tag= 1)
   tree.append(parent= 3, x= 8.0, y= 0.0, z= 0.0, radius= 0.3, tag= 1)
   tree.append(parent= 4, x=10.0, y= 0.0, z= 0.0, radius= 0.5, tag= 1)

.. figure:: gen-images/tree2c.svg
  :width: 300
  :align: center

  Samples 3 and 4 are collocated with different radii.

.. code:: Python

    morph = arbor.morphology(tree, spherical_root=False)

.. figure:: gen-images/morph2c.svg
  :width: 600
  :align: center

  The resulting morphology has a step discontinuity in radius.

.. _morph-example4:

Example 3: Y-shaped cell
""""""""""""""""""""""""""""""

The simplest branching morphology is a cable that bifurcates into two branches,
which we call a *y-shaped cell*.
In the example below, the first branch of the tree is a cable of length 10 μm with a
a radius that tapers from 0.5 μm to 1 μm.
The two child branches are attached to the end of the first branch, and taper from from 0.5 μ m
to 0.2 μm.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 1.0, tag= 1)
   tree.append(parent= 0, x=10.0, y= 0.0, z= 0.0, radius= 0.5, tag= 1)
   tree.append(parent= 1, x=15.0, y= 3.0, z= 0.0, radius= 0.2, tag= 1)
   tree.append(parent= 1, x=15.0, y=-3.0, z= 0.0, radius= 0.2, tag= 1)

.. figure:: gen-images/tree3a.svg
  :width: 400
  :align: center

.. code:: Python

   morph = arbor.morphology(tree, spherical_root=False)

.. figure:: gen-images/morph3a.svg
  :width: 800
  :align: center


The child branches above start with the same radius of 0.5 μm as the distal end of their parent branch.
For the children to have a constant radius of 0.2 μm, instead of tapering from 0.5 μm to 0.2 μm,
we use collocated samples of radius 0.2 μm.
Two methods that use the same approach are illustrated below:

* insert collocated points at the start of each child branch;
* insert a single collocated point at the end of the parent branch.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 1.0, tag= 1)
   tree.append(parent= 0, x=10.0, y= 0.0, z= 0.0, radius= 0.5, tag= 1)
   tree.append(parent= 1, x=10.0, y= 0.0, z= 0.0, radius= 0.2, tag= 1)
   tree.append(parent= 2, x=15.0, y= 3.0, z= 0.0, radius= 0.2, tag= 1)
   tree.append(parent= 1, x=10.0, y= 0.0, z= 0.0, radius= 0.2, tag= 1)
   tree.append(parent= 4, x=15.0, y=-3.0, z= 0.0, radius= 0.2, tag= 1)

.. figure:: gen-images/tree3b.svg
  :width: 400
  :align: center

  The first approach has 3 collocated points at the fork: sample 1 is at the end
  of the parent branch, and samples 2 and 4 are attached to sample 1 and are at
  the start of the children branches.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 1.0, tag= 1)
   tree.append(parent= 0, x=10.0, y= 0.0, z= 0.0, radius= 0.5, tag= 1)
   tree.append(parent= 1, x=10.0, y= 0.0, z= 0.0, radius= 0.2, tag= 1)
   tree.append(parent= 2, x=15.0, y= 3.0, z= 0.0, radius= 0.2, tag= 1)
   tree.append(parent= 2, x=15.0, y=-3.0, z= 0.0, radius= 0.2, tag= 1)

.. figure:: gen-images/tree3c.svg
  :width: 400
  :align: center

  The second approach has 2 collocated points at the fork.
  The first collocated sample with radius 0.5 μm is the end of the parent branch, and
  both child branches connect to the second sample with radius to 0.2 μm.

.. code:: Python

   morph = arbor.morphology(tree, spherical_root=False)

.. figure:: gen-images/morph3b.svg
  :width: 800
  :align: center

  The resulting morphology is the same for both approaches.

.. _morph-tree5:

Example 4: Ball and stick
""""""""""""""""""""""""""""""

The next example is a spherical soma of radius 3 μm with a single branch of length
7 μm and constant radius of 1 μm attached.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 2.0, tag= 1)
   tree.append(parent= 0, x= 2.0, y= 0.0, z= 0.0, radius= 1.0, tag= 1)
   tree.append(parent= 1, x=10.0, y= 0.0, z= 0.0, radius= 1.0, tag= 1)

.. figure:: gen-images/tree4.svg
  :width: 300
  :align: center

This sample tree has three points that are connected together in a line, and could
be interpreted as a single unbranched cable.

.. code:: Python

   morph = arbor.morphology(tree, spherical_root=False)

.. figure:: gen-images/morph4a.svg
  :width: 600
  :align: center

To achieve the desired model of a spherical soma with a single cable segment attached,
generate the morphology with ``spherical_root=True``:

.. code:: Python

   morph = arbor.morphology(tree, spherical_root=True)

.. figure:: gen-images/morph4b.svg
  :width: 600
  :align: center

  The spherical root is a special branch with id 0, and the dendrite is a second branch numbered 1.

Example 5: Branches and soma
"""""""""""""""""""""""""""""""""""""

This example models a cell with a simple dendritic tree attached to a soma.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 3.0, tag= 1)
   tree.append(parent= 0, x= 5.0, y=-1.0, z= 0.0, radius= 0.8, tag= 1)
   tree.append(parent= 1, x=10.0, y= 0.5, z= 0.0, radius= 0.5, tag= 1)
   tree.append(parent= 2, x=15.0, y= 0.0, z= 0.0, radius= 0.5, tag= 1)
   tree.append(parent= 3, x=18.0, y= 5.0, z= 0.0, radius= 0.3, tag= 1)
   tree.append(parent= 4, x=23.0, y= 8.0, z= 0.0, radius= 0.3, tag= 1)
   tree.append(parent= 3, x=20.0, y=-4.0, z= 0.0, radius= 0.3, tag= 1)

The root sample with id 0 has a large radius to represent the soma, and the dendritic
tree is represented by samples 1-6.

.. figure:: gen-images/tree5a.svg
  :width: 400
  :align: center

If the morphology is generated without a spherical root, that is with ``spherical_root=False``,
the soma is treated as a truncated cone whose end points are defined by between samples 0 and 1.

.. code:: Python

   morph = arbor.morphology(tree, spherical_root=False)

.. figure:: gen-images/morph5a_cable.svg
  :width: 800
  :align: center

  **Left**: The entire cell is composed of frustums.
  **Right**: There are three branches, with branch 0 containing both the soma and the first dendrite.

If the first sample is treated as a spherical soma by setting ``spherical_root=True``, the
morphology has 4 branches, with the soma having its own spherical branch, and the dendritic tree
composed of 3 branches.

.. code:: Python

   morph = arbor.morphology(tree, spherical_root=True)

.. figure:: gen-images/morph5a_sphere.svg
  :width: 800
  :align: center

If the morphology is meant to model a cell with a spyherical soma, an additional sample can be added at
the edge of the soma to bridge the gap and "fix" the cell.

.. code:: Python

   tree = arbor.sample_tree()
   tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 3.0, tag= 1)
   tree.append(parent= 0, x= 3.0, y=-0.8, z= 0.0, radius= 0.8, tag= 1)
   tree.append(parent= 1, x= 5.0, y=-1.0, z= 0.0, radius= 0.8, tag= 1)
   tree.append(parent= 2, x=10.0, y= 0.5, z= 0.0, radius= 0.5, tag= 1)
   tree.append(parent= 3, x=15.0, y= 0.0, z= 0.0, radius= 0.5, tag= 1)
   tree.append(parent= 4, x=18.0, y= 5.0, z= 0.0, radius= 0.3, tag= 1)
   tree.append(parent= 5, x=23.0, y= 8.0, z= 0.0, radius= 0.3, tag= 1)
   tree.append(parent= 4, x=20.0, y=-4.0, z= 0.0, radius= 0.3, tag= 1)

.. figure:: gen-images/tree5b.svg
  :width: 400
  :align: center

  Sample tree with an additional sample added to the surface of the spherical root.

.. code:: Python

   morph = arbor.morphology(tree, spherical_root=True)

.. figure:: gen-images/morph5b_sphere.svg
  :width: 800
  :align: center

  The morphology has no gap between the soma and the start of the dendritic tree.

