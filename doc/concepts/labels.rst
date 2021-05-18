.. _labels:

Cable cell labels
=================

Arbor provides a domain specific language (DSL) for describing :term:`regions <region>` and
:term:`locations <locset>` on :term:`morphologies <morphology>`, and a :ref:`dictionary <labels-dictionary>` for associating these :ref:`expressions <labels-expressions>`
with a string :term:`label`.

The labels are used to refer to regions
and locations when setting cell properties and attributes.
For example, the membrane capacitance on a region of the cell membrane, or
the location of synapse instances.

Example cell
------------

The following morphology is used on this page to illustrate region and location
descriptions. It has a soma, dendritic tree and an axon with a hillock:

.. _labels-morph-fig:

.. figure:: ../gen-images/label_morph.svg
  :width: 800
  :align: left

  Segments of the morphology are coloured according to tags:
  soma (tag 1, red), axon (tag 2, grey), dendrites (tag 3, blue) (left).
  The 6 branches of the morphology with their branch ids (right).

*Branch 0* contains the soma, which is modelled as a cylinder of length and diameter 4 μm,
and the proximal unbranched section of the dendritic tree which has a radius of 0.75 μm,
attached to the distal end of the soma.

The other branches in the dendritic tree have the following properties:

* *branch 1* tapers from 0.4 to 0.2 μm;
* *branch 2* has a constant radius of 0.5 μm;
* *branch 3* tapers from 0.5 to 0.2 μm;
* *branch 4* tapers from 0.5 to 0.2 μm.

*Branch 5* is the axon, composed of two cable segments: an axon hillock with a radius that
tapers from 4 μm to 0.4 μm attached to the proximal end of the soma; and the start of the
axon proper with constant radius 0.4 μm.

Label types
-----------

.. glossary::
  locset
    A locset is a set of :term:`locations <mlocation>` on a morphology, specifically a `multiset <https://en.wikipedia.org/wiki/Multiset>`_,
    which may contain multiple instances of the same location.

Possible locsets might refer to:

* The centre of the soma.
* The locations of inhibitory synapses.
* The tips of the dendritic tree.

.. figure:: ../gen-images/locset_label_examples.svg
  :width: 800
  :align: center

  Examples of locsets on the example morphology.
  The terminal points (left).
  Fifty random locations on the dendritic tree (right).
  The :ref:`root <morph-segment-definitions>` of the morphology is shown with a red circle
  for reference.

.. glossary::

  region
    A region is a subset of a morphology's cable :term:`segments <segment>`.

Some common regions:

* The soma.
* The dendritic tree.
* An explicit reference to a specific unbranched cable, e.g. "branch 3" or "the distal half of branch 1".
* The axon hillock.
* The dendrites with radius less than 1 μm.

It is possible for a region to be empty, for example, a region that defines
the axon will be empty on a morphology that has no axon.

Regions do not need to be complete sub-trees of a morphology, for example,
the region of cables that have radius less than 0.5 μm
:ref:`below <labels-region-examples>` is composed of three disjoint sub-trees.

.. _labels-region-examples:

.. figure:: ../gen-images/region_label_examples.svg
  :width: 800
  :align: center

  Examples of regions on the example morphology. **Left**: The dendritic tree.
  **Right**: All cables with radius less than 0.5 μm.

.. _labels-expressions:

Expressions
-----------

:term:`Regions <region>` and :term:`locsets <locset>` are described using *expressions* written with the DSL.

Examples of expressions that define regions include:

* ``(all)``: the complete cell morphology.
* ``(tag 1)``: all segments with tag 1.
* ``(branch 2)``: branch 2.
* ``(region "soma")``: the region with the label "soma".

Examples of expressions that define locsets include:

* ``(root)``: the location of the :ref:`root points <morph-segment-definitions>`.
* ``(terminal)``: the locations of the :ref:`terminal points <morph-segment-definitions>`.
* ``(location 3 0.5)``: the mid point of branch 3.
* ``(locset "synapse-sites")``: the locset labelled "synapse-sites".

Detailed descriptions for all of the region and locset expression types is
given :ref:`below <labels-expr-docs>`.

.. note::
    The example expressions above will look familiar to readers who have
    use the Lisp programming language. This is because both the DSL and Lisp use
    *s-expressions*, which are a simple way to represent a nested list of data.

    However, the DSL is not a dialect of Lisp, and has very simple semantics
    that are only limited to describing morphology features.

Expressions are *composable*, so that expressions can be constructed
from simple expressions. For example, the expression:

.. code-block:: lisp

    (radius-lt (join (tag 3) (tag 4)) 0.5)

describes the region of all parts of a cell with either tag 3 or tag 4 and radius less than 0.5 μm.

.. note:

    In NEURON *prescriptive* hoc templates are typically used to calculate
    explicit lists of sections or segments using loops and logical constructs.
    The logic in a hoc template often makes it difficult to understand
    what the results describe, and is error prone.

    Arbor expressions are *descriptive*, in that they describe *what* a
    region or locset is, not *how* it is to be computed.
    As a result, label dictionaries are much more concise and easy to interpret for
    consumers of a model than hoc templates.
    Furthermore they are less error prone because
    Arbor handles generation of concrete cable sections and locations when
    expressions are applied to a morphology.

.. _labels-expr-docs:

Expression syntax
~~~~~~~~~~~~~~~~~

The DSL uses `s-expressions <https://en.wikipedia.org/wiki/S-expression>`_, which are composed of the following basic types:

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

Expressions can be written over multiple lines, and comments are marked with semi-colon.
This can be used to make more complex expression easier to read, for example the
following region that finds all the sub-trees that start at the locations on the
dendritic tree where the radius first is less than or equal to 0.2 μm.

.. code:: lisp

    (distal-interval                   ; take subtrees that start at
        (proximal                      ; locations closest to the soma
            (radius-le                 ; with radius <= 0.2 um
                (join (tag 3) (tag 4)) ; on basal and apical dendrites
                0.2)))

.. note::
    If the expression above at first seems a little complex, consider how the same
    thing could be achieved using hoc in NEURON, and whether it would be free of bugs
    and applicable to arbitrary morphologies.

.. _labels-locset-expr:

Locset expressions
~~~~~~~~~~~~~~~~~~

.. figure:: ../gen-images/label_branch.svg
  :width: 800
  :align: center

  The input morphology with branch numbers for reference in the examples below.


.. label:: (root)

    The location of the root.

    Equivalent to ``(location 0 0)``.

    .. figure:: ../gen-images/root_label.svg
      :width: 300
      :align: center

.. _labels-location-def:

.. label:: (location branch:integer pos:real)

    A location on ``branch``, where ``0 ≤ pos ≤ 1`` gives the relative position
    between the proximal and distal ends of the branch. The position is in terms
    of branch length, so for example, on a branch of length 100 μm ``pos=0.2``
    corresponds to 20 μm from the proximal end, or 80 μm from the distal end.

    .. figure:: ../gen-images/location_label.svg
      :width: 300
      :align: center

      The result of ``(location 1 0.5)``, which corresponds to the mid point of branch 1.

.. label:: (terminal)

    The location of terminal points, which are the most distal locations on the morphology.
    These will typically correspond to the tips, or end points, of dendrites and axons.

    .. figure:: ../gen-images/term_label.svg
      :width: 300
      :align: center

      The terminal points, generated with ``(terminal)``.

.. label:: (uniform reg:region first:int last:int seed:int)

    .. figure:: ../gen-images/uniform_label.svg
      :width: 600
      :align: center

      Ten random locations on the dendrites drawn using different random seeds.
      On the left with  a seed of 0: ``(uniform (tag 3) 0 9 0)``,
      and on the right with  a seed of 1: ``(uniform (tag 3) 0 9 1)``.

.. label:: (on-branches pos:double)

    The set of locations ``{(location b pos) | 0 ≤ b < nbranch-1}``.

    .. figure:: ../gen-images/on_branches_label.svg
      :width: 300
      :align: center

      The set of locations at the midpoint of every branch, expressed as ``(on-branches 0.5)``.

.. label:: (distal reg:region)

    The set of the most distal locations of a region.
    These are defined as the locations for which there are no other locations more distal in the region.

    .. figure:: ../gen-images/distal_label.svg
      :width: 600
      :align: center

      On the left is the region with radius between 0.3 μm and 0.5 μm.
      The right shows the distal set of this region.

.. label:: (proximal reg:region)

    The set of the most proximal locations of a region.
    These are defined as the locations for which there are no other locations more proximal in the region.

    .. figure:: ../gen-images/proximal_label.svg
      :width: 600
      :align: center

      On the left is the region with radius between 0.3 μm and 0.5 μm.
      The right shows the proximal set of this region.

.. label:: (locset name:string)

    Refer to a locset by its label. For example, ``(locset "synapse-sites")`` could be used in an expression to refer
    to a locset with the name ``"synapse-sites"``.

.. label:: (restrict locations:locset reg:region)

    The set of locations in the locset ``loc`` that are in the region ``reg``.

    .. figure:: ../gen-images/restrict_label.svg
      :width: 600
      :align: center

      The result of restricting the terminal locations (left) onto the dendritic tree (middle) is the tips of the dendritic tree (right).

      .. code-block:: lisp

        (restrict (terminal) (tag 3))


.. label:: (join lhs:locset rhs:locset [...locset])

    Set intersection for two locsets, with duplicates removed and results sorted.
    For example, the following:

    .. code-block:: lisp

        (join
            (join (location 1 0.5) (location 2 0.1) (location 1 0.2))
            (join (location 1 0.5) (location 4 0)))

    Gives the following:

    .. code-block:: lisp

        (join (location 1 0.2) (location 1 0.5) (location 2 0.1) (location 4 0))

    Note that ``(location 1 0.5)`` occurs in both the sets, and occurs only once in the result.

.. label:: (sum lhs:locset rhs:locset [...locset])

    Multiset summation of two locsets, such that ``(sum lhs rhs) = A + B``, where A and B are multisets of locations.
    This is equivalent to concatenating the two lists, and the length of the result is the sum of
    the lengths of the inputs. For example:

    .. code-block:: lisp

        (sum
            (join (location 1 0.5) (location 2 0.1) (location 1 0.2))
            (join (location 1 0.5) (location 4 0)))

    Gives the following:

    .. code-block:: lisp

        (join (location 1 0.5) (location 2 0.1) (location 1 0.2) (location 1 0.5) (location 4 0))

.. _labels-region-expr:

Region expressions
~~~~~~~~~~~~~~~~~~

.. label:: (nil)

    An empty region.

.. label:: (all)

    All branches in the morphology.

    .. figure:: ../gen-images/nil_all_label.svg
      :width: 600
      :align: center

      The trivial region definitions ``(nil)`` (left) and ``(all)`` (right).

.. label:: (tag tag_id:integer)

    All of the segments with :term:`tag` ``tag_id``.

    .. figure:: ../gen-images/tag_label.svg
      :width: 900
      :align: center

      The soma, axon and dendritic tree, selected using ``(tag 1)``, ``(tag 2)``, and ``(tag 3)`` respectively.


.. label:: (branch branch_id:integer)

    Refer to a branch by its id.

    .. figure:: ../gen-images/branch_label.svg
      :width: 600
      :align: center

      Branches 0 and 3, selected using ``(branch 0)`` and ``(branch 3)`` respectively.

.. label:: (segment segment_id:integer)

    Refer to a segment by its id. Note that segment ids depend on the construction
    order of the morphology. Arbor's morphology loaders are stable in this regard.

    .. figure:: ../gen-images/segment_label.svg
      :width: 600
      :align: center

      Segments 0 and 3, selected using ``(segment 0)`` and ``(segment 3)`` respectively.

.. _labels-cable-def:

.. label:: (cable branch_id:integer prox:real dist:real)

    An unbranched cable that is a subset of ``branch``.
    The values of ``0 ≤ prox ≤ dist ≤ 1`` are the relative position
    of the ends of the branch. The positions are in terms
    of branch length, so for example, on a branch of length 100 μm ``prox=0.2, dist=0.8``
    would give a cable that starts and ends 20 μm and 80 μm along the branch
    respectively.

    .. figure:: ../gen-images/cable_label.svg
      :width: 600
      :align: center

      Selecting parts of branch 1, from left to right: ``(cable 1 0 1)`` to select the
      whole branch, ``(cable 1 0.3 1)`` and ``(cable 0 0.3 0.7)`` to select part of the branch.

.. label:: (region name:string)

    Refer to a region by its label. For example, ``(region "axon")`` would refer to a region with the label ``"axon"``.

.. label:: (distal-interval start:locset extent:real)

    The distal interval of a location is the region that contains all points that are distal to the location,
    and up to ``extent`` μm from the location, measured as the distance traversed along cables between two locations.
    The distal interval of the locset ``start`` is the union of the distal interval of each location in ``start``.

    .. figure:: ../gen-images/distint_label.svg
      :width: 600
      :align: center

      On the left is a locset of 3 locations: 1 on the axon and 2 in the dendritic tree.
      The right shows the locset's distal interval with extent 5 μm, formed with the following expression:

      .. code-block:: lisp

        (distal-interval (sum (location 1 0.5) (location 2 0.7) (location 5 0.1)) 5)

.. label:: (distal-interval start:locset)

    When no ``extent`` distance is provided, the distal intervals are extended to all terminal
    locations that are distal to each location in ``start``.

    .. figure:: ../gen-images/distintinf_label.svg
      :width: 600
      :align: center

      On the left is a locset of 3 locations: 1 on the axon and 2 in the dendritic tree.
      The right shows the locset's distal interval formed with the following expression:

      .. code-block:: lisp

        (distal-interval (sum (location 1 0.5) (location 2 0.7) (location 5 0.1)))


.. label:: (proximal-interval start:locset extent:real)

    The proximal interval of a location is the region that contains all points that are proximal to the location,
    and up to ``extent`` μm from the location, measured as the distance traversed along cables between two locations.
    The proximal interval of the locset ``start`` is the union of the proximal interval of each location in ``start``.

    .. figure:: ../gen-images/proxint_label.svg
      :width: 600
      :align: center

      On the left is a locset with two locations on separate sub-trees of the dendritic tree.
      On the right is their proximal interval with an ``extent`` of 5 μm, formed as follows:

      .. code-block:: lisp

        (proximal-interval (sum (location 1 0.8) (location 2 0.3)) 5)

.. label:: (proximal-interval start:locset)

    When no ``extent`` distance is provided, the proximal intervals are extended to the root location.

    .. figure:: ../gen-images/proxintinf_label.svg
      :width: 600
      :align: center

      On the left is a locset with two locations on separate sub-trees of the dendritic tree.
      On the right is their proximal interval formed as follows:

      .. code-block:: lisp

        (proximal-interval (sum (location 1 0.8) (location 2 0.3)))

.. label:: (radius-lt reg:region radius:real)

    All parts of cable segments in the region ``reg`` with radius less than ``radius``.

    .. figure:: ../gen-images/radiuslt_label.svg
      :width: 300
      :align: center

      All cable segments with radius **less than** 0.5 μm, found by applying ``radius-lt`` to all of
      the cables in the morphology.
      Note that branch 2, which has a constant radius of 0.5 μm, is not in the result because its radius
      is not strictly less than 0.5 μm.

      .. code-block:: lisp

        (radius-lt (all) 0.5)

.. label:: (radius-le reg:region radius:real)

    All parts of cable segments in the region ``reg`` with radius less than or equal to ``radius``.

    .. figure:: ../gen-images/radiusle_label.svg
      :width: 300
      :align: center

      All cable segments with radius **less than or equal to** 0.5 μm, found by applying ``radius-le`` to all of
      the cables in the morphology.
      Note that branch 2, which has a constant radius of 0.5 μm, is in the result.

      .. code-block:: lisp

        (radius-le (all) 0.5)

.. label:: (radius-gt reg:region radius:real)

    All parts of cable segments in the region ``reg`` with radius greater than ``radius``.

    .. figure:: ../gen-images/radiusgt_label.svg
      :width: 300
      :align: center

      All cable segments with radius **greater than** 0.5 μm, found by applying ``radius-ge`` to all of
      the cables in the morphology.
      Note that branch 2, which has a constant radius of 0.5 μm, is not in the result because its radius
      is not strictly greater than 0.5 μm.

      .. code-block:: lisp

        (radius-gt (all) 0.5)

.. label:: (radius-ge reg:region radius:real)

    All parts of cable segments in the region ``reg`` with radius greater than or equal to ``radius``.

    .. figure:: ../gen-images/radiusge_label.svg
      :width: 300
      :align: center

      All cable segments with radius **greater than or equal to** 0.5 μm, found by applying ``radius-le`` to all of
      the cables in the morphology.
      Note that branch 2, which has a constant radius of 0.5 μm, is in the result.

      .. code-block:: lisp

        (radius-ge (all) 0.5)

.. label:: (join lhs:region rhs:region [...region])

    The union of two or more regions.

    .. figure:: ../gen-images/union_label.svg
      :width: 900
      :align: center

      Two regions (left and middle) and their union (right).

.. label:: (intersect lhs:region rhs:region [...region])

    The intersection of two or more regions.

    .. figure:: ../gen-images/intersect_label.svg
      :width: 900
      :align: center

      Two regions (left and middle) and their intersection (right).

.. _labels-thingify:

Thingification
--------------

When a region or locset expression is applied to a cell morphology, it is represented
as a list of unbranched :term:`cables <cable>` or a set of :term:`locations <mlocation>` on the morphology.
This process is called ``thingify`` in Arbor, because it turns the abstract description
of a :term:`region` or a :term:`locset` into an actual 'thing' when it is applied to a real morphology.

.. note::
    Applying an expression to different morphologies may give different
    thingified results.

.. _labels-locations:

Locations
~~~~~~~~~

A :term:`location <mlocation>` on a cell is described using a tuple ``(branch, pos)``, where
``branch`` is a branch id, and ``0 ≤ pos ≤ 1`` is the relative distance along
the branch, given that 0 and 1 are the proximal and distal ends of the branch
respectively.

Examples of locations, :ref:`expressed using the DSL <labels-location-def>`, include:

* The root ``(location 0 0)``.
* The start of branch 5 ``(location 5 0)``.
* The end of branch 5 ``(location 5 1)``.
* One quarter of the way along branch 5 ``(location 5 0.25)``.

In general, a location on a component can be specific with ``on-components``, e.g.:

* One quarter of the way along segment 3 ``(on-components 0.25 (segment 3))``.
* One tenth of the way along branch 4 ``(on-components 0.1 (branch 4))`` (identical to ``(location 4 0.1)``).

.. _labels-cables:

Cables
~~~~~~

An unbranched :term:`cable` is a tuple of the form ``(branch, prox, dist)``,
where ``branch`` is the branch id, and ``0 ≤ prox ≤ dist ≤ 1`` define the relative position
of the end points of the section on the branch.

Examples of cables, :ref:`expressed using the DSL <labels-cable-def>`, include:

* All of branch 2 ``(cable 2 0 1)``.
* The middle third of branch 2 ``(cable 2 0.333 0.667)``.
* A zero length cable in the middle of branch 2 ``(cable 2 0.5 0.5)``.

.. note::
    Zero length cables are permitted.
    They are not useful for defining membrane properties, which are applied to
    the surface of a region.
    However, they can occur as the result of sub-expressions in larger
    expressions that define non-trivial regions and locsets.

.. _labels-dictionary:

Label Dictionaries
------------------

.. glossary::
  label
    A label is a string assigned to an :ref:`expression <labels-expressions>`, and used to refer to the expression or the
    concrete :term:`region` or :term:`locset` generated when the expression is applied to a morphology.

Although any string is a valid label, it is a good idea to avoid labels that would
also be valid expressions in the region DSL; creating a label ``"(tag 1)"`` will only
lead to confusion.

.. glossary::
  label dictionary
    An Arbor structure in which labels are stored with their associated expressions as key-value pairs.

Label dictionaries are used to create a cable-cell along with the :ref:`morphology <morph>`
and a :ref:`decor <cablecell-decoration>`. The decorations can be painted or placed on
the regions or locsets defined in the label dictionary by referring to their labels.

.. code-block:: python
   :caption: Example of a label dictionary in python:

    arbor.label_dict({
      'soma': '(tag 1)',  # soma is every cable with tag 1 in the morphology.
      'axon': '(tag 2)',  # axon is every cable with tag 2 in the morphology.
      'dend': '(tag 3)',  # dend is every cable with tab 3 in the morphology
      'root': '(root)',   # typically the start of the soma is at the root of the cell.
      'stim_site': '(location 0 0.5)', # site for the stimulus, in the middle of branch 0.
      'axon_end': '(restrict (terminal) (region "axon"))'} # end of the axon.
    })


API
---

* :ref:`Python <pylabels>`
* *TODO*: C++ documentation.
