.. _cpplabels:

Cable cell labels
=================

.. currentmodule:: arbor

.. cpp:class:: label_dict

   Stores labels and their associated :ref:`expressions <labels-expressions>` as key-value pairs.

   .. cpp:function:: label_dict()

   .. cpp:function:: label_dict(const label_dict&)

   .. cpp:function:: label_dict& extend(const label_dict& other, const std::string& prefix = "")

      Add all definitions from ``other``, optionally adding a prefix.
          
   .. cpp:function:: const std::unordered_map<std::string, arb::region>& regions() const

      The region definitions in the dictionary.

   .. cpp:function:: const std::unordered_map<std::string, arb::locset>& locsets() const

      The locset definitions in the dictionary.

   .. cpp:function:: const std::unordered_map<std::string, arb::iexpr>& iexpressions() const

      The iexpr definitions in the dictionary.

   .. cpp:function:: label_dict& set(const std::string& name, locset ls)

      Add locset under ``name``.
      
   .. cpp:function:: label_dict& set(const std::string& name, region reg)

      Add region under ``name``.

   .. cpp:function:: label_dict& set(const std::string& name, iexpr e)

      Add iexpr under ``name``.

   .. cpp:function:: add_swc_tags

      Add SWC default regions

   .. cpp:function:: std::size_t erase(const std::string& key)

      Remove definitions for ``key``.
      
   .. cpp:function:: std::optional<locset> locset(const std::string& name)

      Try to find locset under ``name``.

   .. cpp:function:: std::optional<region> region(const std::string& name)

      Try to find region under ``name``.

   .. cpp:function:: std::optional<iexpr> iexpr(const std::string& name)

      Try to find iexpr under ``name``.

The ``arb::label_dict`` type is used for creating and manipulating label
dictionaries. For example, a dictionary that uses tags that correspond to SWC
`structure identifiers
<http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_
to label soma, axon, basal dendrites, and apical dendrites is:

.. code-block:: cpp

   using namespace arborio::literals;

   auto d = arb::label_dict();

   // same as d.add_swc_tags()
   d.set("soma", arb::reg::tagged(1));
   d.set("axon", arb::reg::tagged(2));
   d.set("dend", arb::reg::tagged(3));
   d.set("apic", arb::reg::tagged(4));

The ``set`` method is used above to add label definitions. It can be
used to modify existing definitions, so long as the new definition has the same
type (region or locset), continuing:

.. code-block:: cpp

    // A label can be overwritten with a definition of the
    // same type, in this case, a region.
    d.set("dend", "(join (tag 3) (tag 4))"_reg);

    // However, a region can"t be overwritten by a locset, or vice-versa.
    d.set("dend", "(terminal)"_ls); error: "(terminal)" defines a locset.

:Ref:`Expressions <labels-expressions>` can refer to other regions and locsets
in a label dictionary. In the example below, we define a new region labeled that
is the union of both the ``dend`` and ``apic`` regions.

.. code-block:: cpp

    // equivalent to (join (tag 3) (tag 4))
    d.set("3+4", arb::reg::join(arb::reg::named("dend"),
                                arb::reg::named("apic")));

The order which labels are defined in does not matter, so an :ref:`expression
<labels-expressions>` can refer to a label that has not yet been defined,
continuing our running example

.. code-block:: cpp

    // If d were applied to a morphology, "reg" would refer to the region:
    //   "(distal_interval (location 3 0.5))"
    // Which is the sub-tree of the matrix starting at "(location 3 0.5)"
    d.set("reg", "(distal_interval (locset \"loc\"))"_ls);
    d.set("loc", "(location 3 0.5)"_ls);

    // The locset "loc" can be redefined
    d.set("loc",  "(proximal (tag 3))"_ls);

    // Now if d were applied to a morphology, "reg" would refer to:
    //   "(distal_interval (proximal (tag 3))"
    // Which is the subtrees that start at the proximal locations of
    // the region "(tag 3)"

Cyclic dependencies are not permitted, as in the following example where
two labels refer to one another:

.. code-block:: cpp

    d.set("reg", "(distal_interval (locset \"loc\"))"_reg);
    d.set("loc", "(proximal (region \"reg\"))"_ls);

.. note::

   In the example above, there will be no error when the label dictionary is
   defined. Instead, there will be an error later when the label dictionary is
   applied to a morphology, and the cyclic dependency is detected when
   thingifying the locations in the locsets and the cable segments in the
   regions.
=======
Cable Cell Morphology Expressions
=================================

Morphology expressions are used in filling the :ref:`decor` and can be point- or
area-like. Both are constructed and manipulated using static functions in the
namespaces ``reg`` and ``ls`` or by using a DSL.

Regions
-------

The ``region`` objects are reified descriptions of area-like locations, usable
in ``paint`` operations.

.. cpp:namespace:: arb

.. cpp:class:: region

    Representation of a region description

.. cpp:function:: region join(region, region)

    Union of two regions.

.. cpp:function:: region intersect(region, region)

    Intersection of two regions.

.. cpp:function:: region complement(region)

    Closed complement of a region.

.. cpp:function:: region difference(region a, region b)

    (Closure of) set difference of two regions.

.. cpp:namespace:: arb::reg

.. cpp:function:: region nil()

    An empty region.

.. cpp:function:: region cable(msize_t, double, double)

    An explicit cable section.

.. cpp:function:: region branch(msize_t)

    An explicit branch.

.. cpp:function:: region tagged(int id)

    Region with all segments with segment tag id.

.. cpp:function:: region segment(int id)

    Region corresponding to a single segment.

.. cpp:function:: region distal_interval(locset start, double distance)

    Region up to `distance` distal from points in `start`.

.. cpp:function:: region proximal_interval(locset end, double distance)

    Region up to `distance` proximal from points in `start`.

.. cpp:function:: region radius_lt(region reg, double r)

    Region with all segments with radius less than/less than or equal to r

.. cpp:function:: region radius_le(region reg, double r)

    Region with all segments with radius less than/less than or equal to r

.. cpp:function:: region radius_gt(region reg, double r)

    Region with all segments with radius greater than/greater than or equal to r

.. cpp:function:: region radius_ge(region reg, double r)

    Region with all segments with radius greater than/greater than or equal to r


.. cpp:function:: region z_dist_from_root_lt(double r)

    Region with all segments with projection less than/less than or equal to r

.. cpp:function:: region z_dist_from_root_le(double r)

    Region with all segments with projection less than/less than or equal to r


.. cpp:function:: region z_dist_from_root_gt(double r)

    Region with all segments with projection greater than/greater than or equal to r

.. cpp:function:: region z_dist_from_root_ge(double r)

    Region with all segments with projection greater than/greater than or equal to r

.. cpp:function:: region all()

    Region with all segments in a cell.

.. cpp:function:: region complete(region)

    Region including all covers of included fork points. (Pre-image of projection onto the topological tree.)

.. cpp:function:: region named(std::string)

    Region associated with a name.

Locsets
-------

Similar to ``region`` objects, ``locset`` s are reified descriptions of
multisets of point-like locations. These are used in ``place`` operations.

.. cpp:namespace:: arb

.. cpp:class:: locset

    Representation of a locset description

.. cpp:function:: locset join(locset, locset)

    Union of two locsets.

.. cpp:function:: locset sum(locset, locset)

    Multiset sum of two locsets.

.. cpp:namespace:: arb::ls

.. cpp:function:: locset location(msize_t branch, double pos)

    Explicit location on morphology.

.. cpp:function:: locset terminal()

    Set of terminal nodes on a morphology.

.. cpp:function:: locset root()

    The root node of a morphology.

.. cpp:function:: locset named(std::string)

    Named locset.

.. cpp:function:: locset nil()

    The null (empty) set.

.. cpp:function:: locset most_distal(region reg)

    Most distal points of a region.

.. cpp:function:: locset most_proximal(region reg)

    Most proximal points of a region.

.. cpp:function:: locset distal_translate(locset ls, double distance)

    Translate locations in locset distance μm in the distal direction

.. cpp:function:: locset proximal_translate(locset ls, double distance)

    Translate locations in locset distance μm in the proximal direction

.. cpp:function:: locset boundary(region reg)

    Boundary points of a region.

.. cpp:function:: locset cboundary(region reg)

    Completed boundary points of a region. (Boundary of completed components.)

.. cpp:function:: locset restrict_to(arb::locset ls, region reg)

    Returns all locations in a locset that are also in the region.

.. cpp:function:: locset segment_boundaries()

    Returns locations that mark the segments.

.. cpp:function:: locset uniform(region reg, unsigned left, unsigned right, uint64_t seed)

    A range `left` to `right` of randomly selected locations with a uniform distribution from region `reg` generated using `seed`

.. cpp:function:: locset on_branches(double pos)

    Proportional location on every branch.

.. cpp:function:: locset on_components(double relpos, region reg)

    Proportional locations on each component: For each component C of the region, find locations L s.t. dist(h, L) = r * max {dist(h, t) | t is a distal point in C}.

.. cpp:function:: locset support(locset)

    Set of locations in the locset with duplicates removed, i.e. the support of the input multiset)

Inhomogeneous Expressions
-------------------------

.. cpp:namespace:: arb

.. cpp:function:: std::string to_string(const iexpr&)

    Convert to string

.. cpp:function:: iexpr operator+(iexpr a, iexpr b)

    Sum of two iexprs

.. cpp:function:: inline iexpr operator-(iexpr a, iexpr b)

    Difference of two iexprs

.. cpp:function:: iexpr operator*(iexpr a, iexpr b)

    Multiplication of two iexprs

.. cpp:function:: iexpr operator/(iexpr a, iexpr b)

    Division of two iexprs

.. cpp:function:: iexpr operator-(iexpr a)

    Negation of iexpr

.. cpp:class:: iexpr

.. cpp:function:: iexpr scalar(double value)

    Convert double to scalar expr type

.. cpp:function:: iexpr pi()

    pi constant

.. cpp:function:: iexpr distance(double scale, locset loc)

    The minimum distance to any point within the locset ``loc``. The scaling
    parameter ``scale`` has unit :math:`{\mu m}^{-1}` and is multiplied by the
    distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr distance(locset loc)

    The minimum distance to any point within the locset ``loc``. The scaling
    parameter ``scale`` has unit :math:`{\mu m}^{-1}` and is multiplied by the
    distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr distance(double scale, region reg)

    The minimum distance to any point within the region ``reg``. The scaling
    parameter ``scale`` has unit :math:`{\mu m}^{-1}` and is multiplied by the
    distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr distance(region reg)

    The minimum distance to any point within the region ``reg``. The scaling
    parameter ``scale`` has unit :math:`{\mu m}^{-1}` and is multiplied by the
    distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr proximal_distance(double scale, locset loc)

    The minimum distance in proximal direction from the points within the locset
    ``loc``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` and
    is multiplied by the distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr proximal_distance(locset loc)

    The minimum distance in proximal direction from the points within the locset
    ``loc``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` and
    is multiplied by the distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr proximal_distance(double scale, region reg)

    The minimum distance in proximal direction from the points within the region
    ``reg``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` and
    is multiplied by the distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr proximal_distance(region reg)

    The minimum distance in proximal direction from the points within the region
    ``reg``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` and
    is multiplied by the distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr distal_distance(double scale, locset loc)

    The minimum distance in distal direction from the points within the locset
    ``loc``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` and
    is multiplied by the distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr distal_distance(locset loc)

    The minimum distance in distal direction from the points within the locset
    ``loc``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` and
    is multiplied by the distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr distal_distance(double scale, region reg)

    The minimum distance in distal direction from the points within the region
    ``reg``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` and
    is multiplied by the distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr distal_distance(region reg)

    The minimum distance in distal direction from the points within the region
    ``reg``. The scaling parameter ``scale`` has unit :math:`{\mu m}^{-1}` and
    is multiplied by the distance, such that the result is unitless, if absent, 1 is assumed.

.. cpp:function:: iexpr interpolation(double prox_value, locset prox_list, double dist_value, locset dist_list)

    Interpolates between the closest point in the proximal direction in locset
    ``prox_loc`` and the closest point in distal direction ``dist_loc`` with the
    assosiated unitless values ``prox_value`` and ``dist_value``. Evaluates to
    zero if no point is located in each required direction.

.. cpp:function:: iexpr interpolation(double prox_value, region prox_list, double dist_value, region dist_list)

    Interpolates between the closest point in the proximal direction in locset
    ``prox_loc`` and the closest point in distal direction ``dist_loc`` with the
    assosiated unitless values ``prox_value`` and ``dist_value``. Evaluates to
    zero if no point is located in each required direction.

.. cpp:function:: iexpr radius(double scale)

    The radius of the cell at a given point multiplied with the ``scale`` parameter
    with unit :math:`{\mu m}^{-1}`.

.. cpp:function:: iexpr radius()

    The radius of the cell at a given point.

.. cpp:function:: iexpr diameter(double scale)

    The diameter of the cell at a given point multiplied with the ``scale`` parameter
    with unit :math:`{\mu m}^{-1}`.

.. cpp:function:: iexpr diameter()

    The diameter of the cell at a given point.

.. cpp:function:: iexpr add(iexpr left, iexpr right)

    Sum of two iexprs

.. cpp:function:: iexpr sub(iexpr left, iexpr right)

    Difference of two iexprs

.. cpp:function:: iexpr mul(iexpr left, iexpr right)

    Multiplication of two iexprs

.. cpp:function:: iexpr div(iexpr left, iexpr right)

    Division of two iexprs

.. cpp:function:: iexpr exp(iexpr value)

    Exponential

.. cpp:function:: iexpr step_right(iexpr value)

    Step-function (right-sided)

.. cpp:function:: iexpr step_left(iexpr value)

    Step-function (left-sided)

.. cpp:function:: iexpr step(iexpr value)

    Step-function

.. cpp:function:: iexpr log(iexpr value)

    Natural logarithm

.. cpp:function:: iexpr named(std::string name)

    Named expression (from ``label_dict``)

Converting between Strings and Locations
----------------------------------------

Both ``region`` and ``locset`` can be converted to string or parsed from a Scheme-like DSL.

.. cpp:namespace:: arborio

.. cpp:function:: parse_label_hopefully<arb::region> parse_region_expression(const std::string& s)

    Parse ``region`` DSL, returns an ``expected`` of the result or the exception

.. cpp:function:: parse_label_hopefully<arb::locset> parse_locset_expression(const std::string& s)

    Parse ``locset`` DSL, returns an ``expected`` of the result or the exception

.. cpp:function:: parse_label_hopefully<arb::iexpr> parse_iexpr_expression(const std::string& s)

    Parse ``iexpr`` DSL, returns an ``expected`` of the result or the exception

.. cpp:namespace:: arborio::literals

.. cpp:function:: arb::locset operator ""_ls(const char* s, std::size_t)

    String-literal for locsets

.. cpp:function:: arb::region operator ""_reg(const char* s, std::size_t)

    String-literal for locsets