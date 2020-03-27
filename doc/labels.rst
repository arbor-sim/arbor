.. _labels:

Labels
=========

Arbor provides a domain specific language (DSL) for labeling regions and
locations on morphologies.
Labels are used to refer to the morphology when setting cell properties and attributes,
for example, the membrane capacitance on a region of the cell membrane,
or the location of synapse instances.

Example Cell
------------

The following morphology will be used to demonstrate labeling.

.. _labels-morph-fig:

.. figure:: gen-images/morphlab.svg
  :width: 800
  :align: center

  **Left**: Segments of the sample morphology, colored according to tags: soma (tag 1, red), axon (tag 2, gray), dendrites (tag 3, blue).
  **Right**: Branches of the sample morphology.

Branch 0 contains the soma, which is modelled as a cylinder of length and diameter 4 μm, and the first branch of the dendritic tree which has a radius of 0.75 μm, and is attached to the distal end of the soma.
The other branches in the dendritic tree have the following properties: branch 1 tapers from 0.4 to 0.2 μm; branch 2 has radius 0.5 μm;
and branches 3 and 4 taper from 0.5 to 0.2 μm.
The axon is a single branch, composed of two cable segments: a tapering axon hillock attached to the proximal end of the soma, and the start of the axon proper with radius 0.4 μm.

Label Types
------------

Locsets
~~~~~~~~~~~

A *location* is used to place countable entities on the morphology.
Examples of countable entities include synapses, gap junction
sites, voltage recorders and current clamps.
A *locset* is a set of locations.

.. figure:: gen-images/locset_label_examples.svg
  :width: 800
  :align: center

  Examples of locsets on the example morphology. **Left**: The terminal samples.
  **Right**: 50 random locations on the dendritic tree.
  **The root of the morphology is hilighted with a red circle for reference**.


Regions
~~~~~~~~~~~~

A *region* is a subset of a morphology.
Regions are used to define membrane properties, for example the distribution and properties
of ion channels, membrane capacitance, and initial reversal potential.
Examples of regions include:

* The soma.
* The dendrites.
* An explicit reference to a specific unbranched cable, e.g. "branch 3" or "the distal half of branch 1".
* The axon hillock.
* The dendrites with radius less than 1 μm.

It is possible for a region to be empty, for example a region that defines the axon be empty on a morphology that has no axon.
Regions do not need to be complete sub-trees of a morphology.

.. figure:: gen-images/region_label_examples.svg
  :width: 800
  :align: center

  Examples of regions on the example morphology. **Left**: The dendritic tree.
  **Right**: All cables with radius less than 0.5 μm.

Label Dictionaries
------------------

Expressions
~~~~~~~~~~~

Regions and locsets are described using *expressions*, which are written
in a simple s-expression based language.

Examples of expressions that define regions include:

* ``(all)``: the complete cell morphology.
* ``(tag 1)``: all segments with tag 1.
* ``(branch 2)``: branch 2.
* ``(region "soma")``: the region with the label "soma".

And here are examples of expressions that define locsets:

* ``(root)`` -> the root sample.
* ``(terminal)`` -> the terminal samples.
* ``(location 3 0.5)`` -> the mid point of branch 3.
* ``(locset "synapse_sites")`` -> the locset named "synapse_sites".

Detailed descriptions for all of the region and locset expression types is
given :ref:`below <labels-expr-docs>`.

Expressions are *composable*, so that more complex definitions can be constructed
using simple expressions like the examples for regions and locsets above.
For example, the expression
``(radius_lt (join (tag 3) (tag 4)) 0.5)`` describes the region of all parts of a cell
with either tag 3 or tag 4 and radius less than 0.5 μm.

.. note:

    In a typical NEURON workflow, a *prescriptive* hoc template calculates
    explicit lists of sections or segments using loops and logical constructs.
    The logic in a hoc template often makes it difficult to understand
    what the results describe, and is error prone.

    Arbor expressions are *descriptive*, in that they describe *what* a
    region or locset is, not *how* it is to be computed.
    As a result, label dictionaries are much more concise and easy to interpret for
    consumers of a model than hoc templates.
    Furthermore they are less error prone because
    Arbor handles generation of conrete cable sections and locations when
    expressions are applied to a morphology.


Dictionaries
~~~~~~~~~~~~

*Labels* can be assigned to expressions, and used to refer to the expression or the
concrete region or locset generated when the expression is applied to a morphology.
A label is a string with the following rules:

* may contain alpha-numeric values, ``{a-z}[A-z][0-9]``, and underscore ``_`` and hyphen ``-``.
* no leading underscore, hyphen or numeric values: for example `_myregion`, `-samples`, and ``2ndpoint`` are invalid labels.
* no leading numeric values

labels are stored with their associated expressions as key-value pairs in *label dictionaries*.

Representation
----------------

A *location* on a cell is described using a tuple ``(branch, pos)`` where ``branch`` is a
branch id, and ``0 ≤ pos ≤ 1`` is a relative distance along the branch, where 0 and 1 are the
proximal and distal ends of the branch respectively.

Regions are composed of unbranched *cables*, which are tuples of the form ``(branch, prox, dist)``,
where ``branch`` is the branch id, and ``0 ≤ prox ≤ dist ≤ 1`` define the relative position
of the end points of the section on the branch.

*TODO* some examples of cables and locations. (whole branch, subset of branch, root (0,0), mid point of a dendrite)

*TODO* introduce *locset* and *cable_list*

.. _labels-expr-docs:

Expressions Definititions
~~~~~~~~~~~~~~~~~~~~~~~~~

Definition of s expressions

.. generic:: string

    A string literal enclosed in quotes, e.g. ``"dendrites"``.

.. generic:: integer

    An integer. e.g: ``42``, ``-2``, ``0``.

.. generic:: real

    A floating point value. e.g: ``2``, ``4.3``, ``.3``, ``-2.1e3``.

.. generic:: region

    An expression that evaluates to a region. e.g. ``(all)``, ``(tag 3)``, ``(intersect (tag 3) (tag 4))``.

.. generic:: locset

    An expression that evaluates to a locset. e.g. ``(root)``, ``(location 3 0.2)``, ``(proximal (tag 2))``.


Locset Expressions
~~~~~~~~~~~~~~~~~~~~~

.. figure:: gen-images/morphlab.svg
  :width: 800
  :align: center

  The input morphology with branch numbers for reference in the examples below.


.. label:: (root)

    The location of the root sample.

    Equivalent to ``(location 0 0)`` and ``(sample 0)``.

    .. figure:: gen-images/root_label.svg
      :width: 300
      :align: center

.. label:: (location branch:integer pos:real)

    A location on ``branch``, where ``0 ≤ pos ≤ 1`` gives the relative position
    between the proximal and distal ends of the branch. The position is in terms
    of branch length, so for example, on a branch of length 100 μm ``pos=0.2``
    corresponds to 20 μm from the proximal end, or 80 μm from the distal end.

    .. figure:: gen-images/location_label.svg
      :width: 300
      :align: center

      The result of ``(location 1 0.5)``, which corresponds to the mid point of branch 1.

.. label:: (terminal}

    The location of terminal samples, which are the tips, or end points, of dendrites and axons.

    .. figure:: gen-images/term_label.svg
      :width: 300
      :align: center

.. label:: (sample sample_id:integer)

    The location of sample with the id ``sample_id``.

.. label:: (proximal reg:region)

    .. figure:: gen-images/prox1.svg
      :width: 600
      :align: center


    .. figure:: gen-images/prox2.svg
      :width: 600
      :align: center

.. label:: (distal reg:region)

    ``(location 0 0)`` in the distal set below is not prima facie logical: ``(0 0)`` is proximal to ``(1 1)``.
    I can see how we arrived at this result:
    a zero-length cable ``(5 0 0)`` is in the cover, so there is a distal point at ``(5 0)``,
    which gets canonicalised to ``(0 0)``.

    .. figure:: gen-images/dist1.svg
      :width: 600
      :align: center

      (left) the region ``(join (branch 0) (branch 1))``. (right) The result of applying ``distal``: ``((location 0 0) (location 1 1))``.

    That this is some mental gymnastics to justify having the root in the distal set of
    the axon below. I believe that it is reasonable for
    a user who wants to attach a probe at the end of the axon to expect ``(distal (tag 2))`` to give them what
    they expect (I expected it to work that way).
    More worryingly, I can't think of an expression that would give that point, besides an explicit reference ``(location 5 1)``.
    Something like ``(intersect (terminal) (distal (tag 2)))`` would work if we added an ``intersect`` operation
    for locsets, however:

    * that only works because we have a convenient set that contains the point the user is looking for.
    * what a mouthful!

    To address this, I would like to propose adding some new operators that use the canonicalised cover.

    .. figure:: gen-images/dist2.svg
      :width: 600
      :align: center

      (left) the region ``(tag 2)``. (right) The result of applying ``distal``: ``((location 0 0) (location 5 1))``.

   The results of this hold no surprises by virtue of not having a region end at the root.

    .. figure:: gen-images/dist3.svg
      :width: 600
      :align: center

      (left) the region ``((intersect (radius_le (all) 0.5) (radius_ge (all) 0.3)))``.
      (right) The result of applying ``distal``: ``(1 0.793), (3 0.667)  (4 0.391), (5 1))``.

.. label:: (uniform reg:region, first:int, last:int, seed:int)

    .. figure:: gen-images/uniform_label.svg
      :width: 600
      :align: center

      The of drawing 9 random locations on the dendrites using different random seeds:
      ``(uniform (tag 3) 0 9 0)`` (left) and ``(uniform (tag 3) 0 9 1)`` (right).

.. label:: (on_branches pos:double)

.. label:: (locset name:string)

.. label:: (join locset locset [...locset])

.. label:: (sum locset locset [...locset])


Region Expressions
~~~~~~~~~~~~~~~~~~~~~


