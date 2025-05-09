.. _cpplabels:

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
