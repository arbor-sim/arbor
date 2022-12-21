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

.. glossary::
  iexpr
    An iexpr is an inhomogeneous expression, that can be evaluated at any point on a cell.

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

.. label:: (locset-nil)

    The empty locset.

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

    .. figure:: ../gen-images/location_05_label.svg
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

.. label:: (proximal-translate ls:locset distance:real)

    The set of locations that correspond to moving each location in the ``ls`` in the proximal direction
    ``distance`` μm. The locations in the output have a one to one correspondence with those in ``ls``.

    .. figure:: ../gen-images/proximal_translate_label.svg
      :width: 600
      :align: center

      The proximal translation of the terminal locations (left) a distance of 10 μm using the
      expression ``(proximal-translate (terminal) 10)``.

.. label:: (distal-translate ls:locset distance:real)

    The set of locations that correspond to translating each location in ``ls`` in the distal direction
    ``distance`` μm or to a terminal location, whichever is closest.

    An input location will generate multiple output locations when it is translated
    past a fork point, with a new location for each child branch (see the example
    below). For this reason there is not a one-to-one correspondence between locations
    in the input and output sets, so the results are sorted and duplicates are removed.


    .. figure:: ../gen-images/distal_translate_label.svg
      :width: 600
      :align: center

      Two distal translations of the midpoint of branch 0 (left).
      The first translation of 5 μm, ``(distal-translate (location 0 0.5) 5)``, generates a
      single location on the same branch (center).
      The second translation of 15 μm ``(distal-translate (location 0 0.5) 15)`` extends beyond
      the end of branch 0, generating an additional location at each fork point (right).

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

.. label:: (region-nil)

    An empty region.

.. label:: (all)

    All branches in the morphology.

    .. figure:: ../gen-images/nil_all_label.svg
      :width: 600
      :align: center

      The trivial region definitions ``(region-nil)`` (left) and ``(all)`` (right).

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


.. _labels-iexpr:

Inhomogeneous Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. label:: (scalar value:real)

    A scalar of given value.

.. label:: (pi)

    A scalar expression representing the pi constant.

.. label:: (distance scale:real loc:locset)

    The minimum distance to points within the locset ``loc``. The scaling parameter  ``scale`` has unit :math:`{\mu m}^{-1}` 
    and is multiplied with the distance, such that the result is unitless.

    .. figure:: ../images/iexpr_distance.svg
      :width: 600
      :align: center

      The distance between any two points (the evaluation location and a location within the locset), is calculated **along** the entire tree, even across the root.
      Therefore, a distance expression is defined on the entire cell and only zero if evaluated at a location within the locset (or the scale parameter is set to zero).

.. label:: (distance loc:locset)

    A distance expression with a default scaling factor of 1.0.

.. label:: (distance scale:real reg:region)

    The minimum distance to the region ``reg``. Evaluates to zero within the region. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` 
    and is multiplied with the distance, such that the result is unitless.

.. label:: (distance reg:region)

    A distance expression with a default scaling factor of 1.0.

.. label:: (proximal-distance scale:real loc:locset)

    The minimum distance in proximal direction from the points within the locset ``loc``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` 
    and is multiplied with the distance, such that the result is unitless.

    .. figure:: ../gen-images/iexpr_prox_dis.svg
      :width: 600
      :align: center

      Example of a proximal-distance expression with a single input location. **Left**: Input location. **Right**: Area where the expression evaluates to non-zero values.

.. label:: (proximal-distance loc:locset)

    A proximal-distance expression with a default scaling factor of 1.0.

.. label:: (proximal-distance scale:real reg:region)

    The minimum distance in proximal direction from the region ``reg``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` 
    and is multiplied with the distance, such that the result is unitless.

.. label:: (proximal-distance reg:region)

    A proximal-distance expression with a default scaling factor of 1.0.

.. label:: (distal-distance scale:real loc:locset)

    The minimum distance in distal direction from the points within the locset ``loc``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` 
    and is multiplied with the distance, such that the result is unitless.

    .. figure:: ../gen-images/iexpr_dist_dis.svg
      :width: 600
      :align: center

      Example of a distal-distance expression with a single input location. **Left**: Input location. **Right**: Area, where the expression evaluates to non-zero values.

.. label:: (distal-distance loc:locset)

    A distal-distance expression with a default scaling factor of 1.0.

.. label:: (distal-distance scale:real reg:region)

    The minimum distance in distal direction from the region ``reg``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` 
    and is multiplied with the distance, such that the result is unitless.

.. label:: (distal-distance reg:region)

    A distal-distance expression with a default scaling factor of 1.0.

.. label:: (interpolation prox_value:real prox_loc:locset dist_value:real dist_loc:locset)

    Interpolates between the closest point in proximal direction in locset ``prox_loc`` and the closest point in 
    distal direction ``dist_loc`` with the assosiated unitless values ``prox_value`` and ``dist_value``.
    Evaluates to zero, if no point is located in each required direction.

    **Note**: At any fork, an interpolation expression may be discontinuous, if the distance to the closest location within the distal locset differs along each attached branch.

    .. figure:: ../images/iexpr_interp.svg
      :width: 600
      :align: center

      Example of an interpolation expression. **Red**: The root of the morphology. **Blue**: The proximal locset consisting of a single location.
      **Green**: The distal locset, consisting of four locations. Given these locsets, an interpolation expression only evaluates to non-zero in the highlighted area.
      For locations 3 and 4 of the distal locset, there is no location within the proximal locset, that is between them and the root (in proximal direction),
      and therefore an interpolation expression cannot be evaluated and defaults to zero.
      Contrary, for locations 1 and 2 of the distal locset, there is a location within the proximal locset in proximal direction.


.. label:: (interpolation prox_value:real prox_reg:region dist_value:real dist_reg:region)

    Interpolates between the region ``prox_reg`` in proximal diretion and the region ``dist_reg`` in distal direction
    with the associated unitless values ``prox_value`` and ``dist_value``. If evaluated inside either region, returns the corresponding value.
    Evaluates to zero, if no region is located in each required direction.

.. label:: (radius scale:real)

    The radius of the cell at a given point multiplied with the ``scale`` parameter with unit :math:`{\mu m}^{-1}`.

.. label:: (radius)

    A radius expression with a default scaling factor of 1.0.

.. label:: (diameter scale:real)

    The diameter of the cell at a given point multiplied with the ``scale`` parameter with unit :math:`{\mu m}^{-1}`.

.. label:: (diameter)

    A diameter expression with a default scaling factor of 1.0.

.. label:: (add (iexpr | real) (iexpr | real) [... (iexpr | real)])

    Addition of at least two inhomogeneous expressions or real numbers.

.. label:: (sub (iexpr | real) (iexpr | real) [... (iexpr | real)])

    Subtraction of at least two inhomogeneous expressions or real numbers.
    The expression is evaluated from the left to right, subtracting each element from the first one in turn.

.. label:: (mul (iexpr | real) (iexpr | real) [... (iexpr | real)])

    Multiplication of at least two inhomogeneous expressions or real numbers.

.. label:: (div (iexpr | real) (iexpr | real) [... (iexpr | real)])

    Division of at least two inhomogeneous expressions or real numbers.
    The expression is evaluated from the left to right, dividing the first element by each divisor in turn.

.. label:: (exp value:(iexpr | real))

    The exponential function of the inhomogeneous expression or real ``value``.

.. label:: (step_right value:(iexpr | real))

    The Heaviside step function of the inhomogeneous expression or real ``value``, with `(step 0.0)` evaluating to 1.

.. label:: (step_left value:(iexpr | real))

    The Heaviside step function of the inhomogeneous expression or real ``value``, with `(step 0.0)` evaluating to 0.

.. label:: (step value:(iexpr | real))

    The Heaviside step function of the inhomogeneous expression or real ``value``, with `(step 0.0)` evaluating to 0.5.

.. label:: (log value:(iexpr | real))

    The logarithm of the inhomogeneous expression or real ``value``.



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
    concrete :term:`region` or :term:`locset` or :term:`iexpr` generated when the expression is applied to a morphology.

Although any string is a valid label, it is a good idea to avoid labels that would
also be valid expressions in the region DSL; creating a label ``"(tag 1)"`` will only
lead to confusion.

.. glossary::
  label dictionary
    An Arbor structure in which labels are stored with their associated expressions as key-value pairs.

Label dictionaries are used to create a cable-cell along with the :ref:`morphology <morph>`
and a :ref:`decor <cablecell-decoration>`. The decorations can be painted or placed on
the regions, locsets or iexpr defined in the label dictionary by referring to their labels.

.. code-block:: python
   :caption: Example of a label dictionary in python:

    arbor.label_dict({
      'soma': '(tag 1)',  # soma is every cable with tag 1 in the morphology.
      'axon': '(tag 2)',  # axon is every cable with tag 2 in the morphology.
      'dend': '(tag 3)',  # dend is every cable with tab 3 in the morphology
      'root': '(root)',   # typically the start of the soma is at the root of the cell.
      'stim_site': '(location 0 0.5)', # site for the stimulus, in the middle of branch 0.
      'axon_end': '(restrict (terminal) (region "axon"))',  # end of the axon.
      'rad_expr': '(radius 0.5)'  # iexpr evaluating the radius scaled by 0.5
    })


API
---

* :ref:`Python <pylabels>`
* :ref:`C++ <cpplabels>`
