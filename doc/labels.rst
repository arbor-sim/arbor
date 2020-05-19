.. _labels:

Labels
=========

Arbor provides a domain specific language (DSL) for describing regions and
locations on morphologies, and a dictionary for assiciating these descriptions
with a string label.

The labels are used later in the cell building process to refer to regions
and locations when setting cell properties and attributes.
For example, the membrane capacitance on a region of the cell membrane, or
the location of synapse instances.

Example Cell
------------

The following morphology is used on this page to illustrate region and location
descriptions. It has a soma, dendritic tree and an axon with a hillock:

.. _labels-morph-fig:

.. figure:: gen-images/morphlab.svg
  :width: 800
  :align: left

  **Left**: Segments of the sample morphology, colored according to tags:
  soma (tag 1, red), axon (tag 2, grey), dendrites (tag 3, blue).
  **Right**: The 6 branches of the morphology with their branch ids.

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

Label Types
------------

Locsets
~~~~~~~~~~~

A *locset* is a set of locations on a morphology, specifically a *multiset*,
which may contain multiple instances of the same location, for example:

* The center of the soma.
* The locations of inhibitory synapses.
* The tips of the dendritic tree.

.. figure:: gen-images/locset_label_examples.svg
  :width: 800
  :align: center

  Examples of locsets on the example morphology. **Left**: The terminal samples.
  **Right**: 50 random locations on the dendritic tree.
  The :ref:`root <morph-sample-definitions>` of the morphology is hilighted with a red circle
  for reference.


Regions
~~~~~~~~~~~~

A *region* is a subset of a morphology's cable segments, for example:

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

.. figure:: gen-images/region_label_examples.svg
  :width: 800
  :align: center

  Examples of regions on the example morphology. **Left**: The dendritic tree.
  **Right**: All cables with radius less than 0.5 μm.

Expressions
-----------

Regions and locsets are described using *expressions* written with the DSL.

Examples of expressions that define regions include:

* ``(all)``: the complete cell morphology.
* ``(tag 1)``: all segments with tag 1.
* ``(branch 2)``: branch 2.
* ``(region "soma")``: the region with the label "soma".

Examples of expressions that define locsets include:

* ``(root)``: the location of the :ref:`root sample <morph-sample-definitions>`.
* ``(terminal)``: the locations of the :ref:`terminal samples <morph-sample-definitions>`.
* ``(location 3 0.5)``: the mid point of branch 3.
* ``(locset "synapse-sites")``: the locset labelled "synapse-sites".

Detailed descriptions for all of the region and locset expression types is
given :ref:`below <labels-expr-docs>`.

.. note::
    The example expressions above will look familiar to readers who have
    use the Lisp programming language. This is because both the DSL and Lisp use
    *s-expressions*, which are a simple way to represent a nested list of data.

Expressions are *composable*, so that expressions can be constructed
from simple expressions. For example, the expression:

.. code-block:: lisp

    (radius_lt (join (tag 3) (tag 4)) 0.5)

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
    Arbor handles generation of conrete cable sections and locations when
    expressions are applied to a morphology.

.. _labels-expr-docs:

Expression Syntax
~~~~~~~~~~~~~~~~~~~~~~~~~

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

    (distal_interval                   ; take subtrees that start at
        (proximal                      ; locations closest to the soma
            (radius_lte                ; with radius <= 0.2 um
                (join (tag 3) (tag 4)) ; on basal and apical dendrites
                0.2)))

.. note::
    If the expression above at first seems a little complex, consider how the same
    thing could be achieved using hoc in NEURON, and whether it would be free of bugs
    and applicable to arbitrary morphologies.


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

.. _labels-location-def:

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

      The terminal points, generated with ``(terminal)``.

.. label:: (sample sample_id:integer)

    The location of sample with the id ``sample_id``.

    .. figure:: gen-images/sample1_label.svg
      :width: 300
      :align: center

      Sample 1, which defines the distal end of the soma, generated with ``(sample 1)``.

.. label:: (uniform reg:region first:int last:int seed:int)

    .. figure:: gen-images/uniform_label.svg
      :width: 600
      :align: center

      Ten random locations on the dendrites drawn using different random seeds.
      On the left with  a seed of 0: ``(uniform (tag 3) 0 9 0)``,
      and on the right with  a seed of 1: ``(uniform (tag 3) 0 9 1)``.

.. label:: (on_branches pos:double)

    The set of locations ``{(location b pos) | 0 ≤ b < nbranch-1}``.

    .. figure:: gen-images/on_branches_label.svg
      :width: 300
      :align: center

      The set of locations at the midpoint of every branch, expressed as ``(on_branches 0.5)``.

.. label:: (distal reg:region)

    The set of the most distal locations of a region.
    These are defined as the locations for which there are no other locations more distal in the region.

    .. figure:: gen-images/distal_label.svg
      :width: 600
      :align: center

      On the left is the region with radius between 0.3 μm and 0.5 μm.
      The right shows the distal set of this region.

.. label:: (proximal reg:region)

    The set of the most proximal locations of a region.
    These are defined as the locations for which there are no other locations more proximal in the region.

    .. figure:: gen-images/proximal_label.svg
      :width: 600
      :align: center

      On the left is the region with radius between 0.3 μm and 0.5 μm.
      The right shows the proximal set of this region.

.. label:: (locset name:string)

    Refer to a locset by its label. For example, ``(locset "synapse-sites")`` could be used in an expression to refer
    to a locset with the name ``"synapse-sites"``.

.. label:: (restrict locations:locset reg:region)

    The set of locations in the locset ``loc`` that are in the region ``reg``.

    .. figure:: gen-images/restrict_label.svg
      :width: 600
      :align: center

      The result of restricting the terminal locations (left) onto the dendritic tree (middle) is the tips of the dendritic tree (right).

      .. code-block:: lisp

        (restrict (terminal) (tag 3))


.. label:: (join lhs:locset rhs:locset [...locset])

    Set intersection for two locsets, with duplicates removed and results sorted.
    For example:

    ``((1 0.5) (2 0.1) (1 0.2)) ∪ ((1 0.5) (4 0)) = ((1 0.2) (1 0.5) (2 0.1) (4 0))``

    The location ``(1 0.5)`` occurs in both the sets, and occurs only once in the result.

.. label:: (sum lhs:locset rhs:locset [...locset])

    Multiset summation of two locsets, such that ``(sum lhs rhs) = A + B``, where A and B are multisets of locations.
    This is equivalent to contactenating the two lists, and the length of the result is the sum of
    the lenghts of the inputs. For example:

    ``((1 0.5) (2 0.1) (1 0.2)) + ((1 0.5) (4 0)) = ((1 0.5) (2 0.1) (1 0.2) (1 0.5) (4 0))``

Region Expressions
~~~~~~~~~~~~~~~~~~~~~

.. label:: (nil)

    An empty region.

.. label:: (all)

    All branches in the morphology.

    .. figure:: gen-images/nil_all_label.svg
      :width: 600
      :align: center

      The trivial region definitions ``(nil)`` (left) and ``(all)`` (right).

.. label:: (tag tag_id:integer)

    All of the segments with tag ``tag_id``.
    See the :ref:`morphology documentation <morph-tags>` for the definition of tags on segments.

    .. figure:: gen-images/tag_label.svg
      :width: 900
      :align: center

      The soma, axon and dendritic tree, selected using ``(tag 1)``, ``(tag 2)``, and ``(tag 3)`` respectively.


.. label:: (branch branch_id:integer)

    Refer to a branch by its id.

    .. figure:: gen-images/branch_label.svg
      :width: 600
      :align: center

      Branches 0 and 3, selected using ``(branch 0)`` and ``(branch 3)`` respectively.

.. _labels-cable-def:

.. label:: (cable branch_id:integer prox:real dist:real)

    .. figure:: gen-images/cable_label.svg
      :width: 600
      :align: center

      Selecting parts of branch 1: ``(cable 1 0 1)`` to select the whole branch, ``(cable 1 0.3 1)`` and ``(cable 0 0.3 0.7)`` to select part of the branch.

.. label:: (region name:string)

    Refer to a region by its label. For example, ``(region "axon")`` would refer to a region with the label ``"axon"``.

.. label:: (distal_interval start:locset extent:real)

    The distal interval of a location is the region that contains all points that are distal to the location,
    and up to ``extent`` μm from the location, measured as the distance traversed along cables between two locations.
    The distal interval of the locset ``start`` is the union of the distal interval of each location in ``start``.

    .. figure:: gen-images/distint_label.svg
      :width: 600
      :align: center

      On the left is a locset of 3 locations: 1 on the axon and 2 in the dendritic tree.
      The right shows the locset's distal interval with extent 5 μm, formed with the following expression:

      .. code-block:: lisp

        (distal_interval (sum (location 1 0.5) (location 2 0.7) (location 5 0.1)) 5)

.. label:: (distal_interval start:locset)

    When no ``extent`` distance is provided, the distal intervals are extended to all terminal
    locations that are distal to each location in ``start``.

    .. figure:: gen-images/distintinf_label.svg
      :width: 600
      :align: center

      On the left is a locset of 3 locations: 1 on the axon and 2 in the dendritic tree.
      The right shows the locset's distal interval formed with the following expression:

      .. code-block:: lisp

        (distal_interval (sum (location 1 0.5) (location 2 0.7) (location 5 0.1)))


.. label:: (proximal_interval start:locset extent:real)

    The proximal interval of a location is the region that contains all points that are proximal to the location,
    and up to ``extent`` μm from the location, measured as the distance traversed along cables between two locations.
    The proximal interval of the locset ``start`` is the union of the proximal interval of each location in ``start``.

    .. figure:: gen-images/proxint_label.svg
      :width: 600
      :align: center

      On the left is a locset with two locations on separate sub-trees of the dendritic tree.
      On the right is their proximal interval with an ``extent`` of 5 μm, formed as follows:

      .. code-block:: lisp

        (proximal_interval (sum (location 1 0.8) (location 2 0.3)) 5)

.. label:: (proximal_interval start:locset)

    When no ``extent`` distance is provided, the proximal intervals are extended to the root location.

    .. figure:: gen-images/proxintinf_label.svg
      :width: 600
      :align: center

      On the left is a locset with two locations on separate sub-trees of the dendritic tree.
      On the right is their proximal interval formed as follows:

      .. code-block:: lisp

        (proximal_interval (sum (location 1 0.8) (location 2 0.3)))

.. label:: (radius_lt reg:region radius:real)

    All parts of cable segments in the region ``reg`` with radius less than ``radius``.

    .. figure:: gen-images/radiuslt_label.svg
      :width: 300
      :align: center

      All cable segments with radius **less than** 0.5 μm, found by applying ``radius_lt`` to all of
      the cables in the morphology.
      Note that branch 2, which has a constant radius of 0.5 μm, is not in the result because its radius
      is not strictly less than 0.5 μm.

      .. code-block:: lisp

        (radius_lt (all) 0.5)

.. label:: (radius_le reg:region radius:real)

    All parts of cable segments in the region ``reg`` with radius less than or equal to ``radius``.

    .. figure:: gen-images/radiusle_label.svg
      :width: 300
      :align: center

      All cable segments with radius **less than or equal to** 0.5 μm, found by applying ``radius_le`` to all of
      the cables in the morphology.
      Note that branch 2, which has a constant radius of 0.5 μm, is in the result.

      .. code-block:: lisp

        (radius_le (all) 0.5)

.. label:: (radius_gt reg:region radius:real)

    All parts of cable segments in the region ``reg`` with radius greater than ``radius``.

    .. figure:: gen-images/radiusgt_label.svg
      :width: 300
      :align: center

      All cable segments with radius **greater than** 0.5 μm, found by applying ``radius_ge`` to all of
      the cables in the morphology.
      Note that branch 2, which has a constant radius of 0.5 μm, is not in the result because its radius
      is not strictly greater than 0.5 μm.

      .. code-block:: lisp

        (radius_gt (all) 0.5)

.. label:: (radius_ge reg:region radius:real)

    All parts of cable segments in the region ``reg`` with radius greater than or equal to ``radius``.

    .. figure:: gen-images/radiusge_label.svg
      :width: 300
      :align: center

      All cable segments with radius **greater than or equal to** 0.5 μm, found by applying ``radius_le`` to all of
      the cables in the morphology.
      Note that branch 2, which has a constant radius of 0.5 μm, is in the result.

      .. code-block:: lisp

        (radius_ge (all) 0.5)

.. label:: (join lhs:region rhs:region [...region])

    The union of two or more regions.

    .. figure:: gen-images/union_label.svg
      :width: 900
      :align: center

      Two regions (left and middle) and their union (right).

.. label:: (intersect lhs:region rhs:region [...region])

    The intersection of two or more regions.

    .. figure:: gen-images/intersect_label.svg
      :width: 900
      :align: center

      Two regions (left and middle) and their intersection (right).

Concretization
----------------

When a region or locset expression is applied to a cell morphology it is
*concretized*. Concretizing a locset will return a set of *locations* on the
morphology, and concretising a region will return a list of unbranched *cables*
on the morphology.

.. note::
    Applying an expression to different morphologies may give different
    concretized results.

Locations
~~~~~~~~~

A *location* on a cell is described using a tuple ``(branch, pos)``, where
``branch`` is a branch id, and ``0 ≤ pos ≤ 1`` is the relative distance along
the branch, given that 0 and 1 are the proximal and distal ends of the branch
respectively.

Examples of locations, :ref:`expressed using the DSL <labels-location-def>`, include:

* The root ``(location 0 0)``.
* The start of branch 5 ``(location 5 0)``.
* The end of branch 5 ``(location 5 1)``.
* One quarter of the way along branch 5 ``(location 5 0.25)``.

Cables
~~~~~~~~~

An unbranched *cable* is a tuple of the form ``(branch, prox, dist)``,
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


Label Dictionaries
------------------

*Labels* can be assigned to expressions, and used to refer to the expression or the
concrete region or locset generated when the expression is applied to a morphology.
A label is a string with the following rules:

* may contain alpha-numeric values, ``{a-z}[A-z][0-9]``, and underscore
  ``_`` and hyphen ``-``.
* no leading underscore, hyphen or numeric values: for example ``_myregion``,
  ``-samples``, and ``2ndpoint`` are invalid labels.

labels are stored with their associated expressions as key-value pairs in *label dictionaries*.

Python API
----------

The ``arbor.label_dict`` type is used for creating and manipulating label dictionaries,
which can be initialised with a dictionary that defines (label, expression)
pairs. For example, a dictionary that uses tags that correspond to SWC
`structure identifiers <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_
to label soma, axon, dendrite and apical dendrites is:


.. code-block:: python

    import arbor

    labels = {'soma': '(tag 1)',
              'axon': '(tag 2)',
              'dend': '(tag 3)',
              'apic': '(tag 4)'}

    d = arbor.label_dict(labels)

Alternatively, start with an empty label dictionary and add the labels and
their definitions one by one:

.. code-block:: python

    import arbor

    d = arbor.label_dict()

    d['soma'] = '(tag 1)'
    d['axon'] = '(tag 2)'
    d['dend'] = '(tag 3)'
    d['apic'] = '(tag 4)'

The square bracket operator is used above to add label definitions. It can
be used to modify existing definitions, so long as the new new definition has the
same type (region or locset):

.. code-block:: python

    import arbor

    # A label dictionary that defines the label "dend" that defines a region.
    d = arbor.label_dict({'dend': '(tag 3)')

    # The definition of a label can be overwritten with a definition of the
    # same type, in this case a region.
    d['dend'] = '(join (tag 3) (tag 4))'

    # However, a region can't be overwritten by a locset, or vice-versa.
    d['dend'] = '(terminal)' # error: '(terminal)' defines a locset.

    # New labels can be added to the dictionary.
    d['soma'] = '(tag 1)'
    d['axon'] = '(tag 2)'

    # Square brackets can also be used to get a label's definition.
    assert(d['soma'] == '(tag 1)')

Expressions can refer to other regions and locsets in a label dictionary.
In the example below, we define a region labeled *'tree'* that is the union
of both the *'dend'* and *'apic'* regions.

.. code-block:: python

    import arbor

    d = arbor.label_dict({
            'soma': '(tag 1)',
            'axon': '(tag 2)',
            'dend': '(tag 3)',
            'apic': '(tag 4)',
            # equivalent to (join (tag 3) (tag 4))
            'tree': '(join (region "dend") (region "apic"))'})

The order that labels are defined does not matter, so an expression can refer to a
label that has not yet been defined:

.. code-block:: python

    import arbor

    d = arbor.label_dict()
    # 'reg' refers 
    d['reg'] = '(distal_interval (locset "loc"))'
    d['loc'] = '(location 3 0.5)'

    # If d was applied to a morphology, 'reg' would refer to the region:
    #   '(distal_interval (location 3 0.5))'
    # Which is the sub-tree of the matrix starting at '(location 3 0.5)'

    # The locset 'loc' can be redefined
    d['loc'] = '(proximal (tag 3))'

    # Now if d was applied to a morphology, 'reg' would refer to:
    #   '(distal_interval (proximal (tag 3))'
    # Which is the subtrees that start at the proximal locations of
    # the region '(tag 3)'

Cyclic dependencies are not permitted, as in the following example where
two labels refer to one another:

.. code-block:: python

    import arbor

    d = arbor.label_dict()
    d['reg'] = '(distal_interval (locset "loc"))'
    d['loc'] = '(proximal (region "reg"))'

    # Error: 'reg' needs the definition of 'loc', which in turn needs the
    # definition of 'reg'.

.. note::
    In the example above there will be no error when the label dictionary is defined.
    Instead, there will be an error later when the label dictionary is applied to
    a morphology, and the cyclic dependency is detected when concretising the locations
    in the locsets and the cable segments in the regions.


The type of an expression, locset or region, is inferred automatically when it is
input into a label dictionary.
Lists of the labels for regions and locsets are available as attributes:

.. code-block:: python

    import arbor

    d = arbor.label_dict({
            'soma': '(tag 1)',
            'axon': '(tag 2)',
            'dend': '(tag 3)',
            'apic': '(tag 4)',
            'site': '(location 2 0.5)',
            'term': '(terminal)'})

    print('regions: ' + ' '.join(d.regions)) # regions: apic axon dend soma
    print('locsets: ' + ' '.join(d.locsets)) # locsets: site term
