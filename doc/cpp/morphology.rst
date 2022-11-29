.. _cppmorphology:

Cable cell morphology
=====================

Cell morphologies are required to describe a :ref:`cppcablecell`.
Morphologies can be constructed from :cpp:type:`segment_trees`, or read from a number of
file formats; see :ref:`cppcablecell-morphology-construction` for details.

Segment tree
------------

A ``segment_tree`` is -- as the name implies -- a set of segments arranged in a
tree structure, ie each segment has exactly one parent and no child is the
parent of any of its ancestors. The tree starts at a *root* segment which has no
parent. Each segment comprises two points in 3d space together with the radii at
these points. The segment's endpoints are called proximal (at the parent's
distal end) and distal (farther from the root).

Segment trees are used to form morphologies which ignore the 3d information
encoded in the segments and just utilise the radii, length, and tree-structure.
Branches in the tree occur where a segment has more than one child. The tree is
constructed by *appending* segments to the tree. Segments are numbered starting
at ``0`` in the order that they are added, with the first segment getting id
``0``, the second segment id ``1``, and so forth. A segment can not be added
before its parent, hence the first segment is always at the root. In this
manner, a segment tree is always guaranteed to be in a correct state, with
consistent parent-child indexing, and with ``n`` segments numbered from ``0`` to
``n-1``. The first parent must be :data:`mnpos`, indicating 'no parent'.


.. cpp:class:: segment_tree


    .. cpp:function:: segment_tree()

        Construct an empty segment tree.

    .. cpp:function:: msize_t append(msize_t parent, const mpoint& prox, const mpoint& dist, int tag)

        Append a segment to the tree. Returns the new parent's id.

    .. cpp:function:: msize_t append(msize_t parent, const mpoint& dist, int tag)

        Append a segment to the tree whose proximal end has the location and
        radius of the distal end of the parent segment. Returns the new
        parent's id.

        This version of append can't be used for a segment at the root of the
        tree, that is, when ``parent`` is :data:`mnpos`, in which case both
        proximal and distal ends of the segment must be specified.

    .. cpp:function:: bool empty()

        If the tree is empty (i.e. whether it has size 0)

    .. cpp:function:: msize_t size()

        The number of segments.

    .. cpp:function:: std::vector<msize_t> parents()

        A list of parent indices of the segments.

    .. cpp:function:: std::vector<msegment> segments()

        A list of the segments.

.. cpp:function:: std::pair<segment_tree, segment_tree> split_at(const segment_tree& t, msize_t id)

    Split a segment_tree into a pair of subtrees at the given id,
    such that one tree is the subtree rooted at id and the other is the
    original tree without said subtree.

.. cpp:function:: segment_tree join_at(const segment_tree& t, msize_t id, const segment_tree& o)

    Join two subtrees at a given id, such that said id becomes the parent
    of the inserted sub-tree.

.. cpp:function:: std::vector<msize_t> tag_roots(const segment_tree& t, int tag)

    Get IDs of roots of a region with specific tag in the segment tree, i.e. segments whose
    parent is either :data:`mnpos` or a segment with a different tag.

.. cpp:function:: bool equivalent(const segment_tree& l, const segment_tree& r)

    Two trees are equivalent if
    1. the root segments' ``prox`` and ``dist`` points and their ``tags`` are identical.
    2. recursively: all sub-trees starting at the current segment are pairwise equivalent.

    Note that under 1 we do not consider the ``id`` field.

.. cpp:function:: segment_tree apply(const segment_tree& t, const isometry& i)

    Apply an :cpp:type:`isometry` to the segment tree, returns the transformed tree as a copy.
    Isometries are rotations around an arbritary axis and/or translations; they can
    be instantiated using ``isometry::translate`` and ``isometry::rotate`` and combined
    using the ``*`` operator.

Morphology API
--------------

.. todo::

   Describe morphology methods.

.. _cppcablecell-morphology-construction:

The stitch-builder interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like the segment tree, the :cpp:type:`stich_builder` class constructs morphologies
through attaching simple components described by a pair of :cpp:type:`mpoint` values,
proximal and distal. These components are :cpp:type:`mstitch` objects, and
they differ from segments in two regards:

1. Stitches are identified by a unique string identifier, in addition to an optional tag value.

2. Stitches can be attached to a parent stitch at either end, or anywhere in the middle.

The ability to attach a stitch some way along another stitch dictates that one
stitch may correspond to more than one morphological segment once the morphology
is fully specified. When these attachment points are internal to a stitch, the
corresponding geometrical point is determined by linearly interpolating between
the proximal and distal points.

The required header file is ``arbor/morph/stitch.hpp``.

:cpp:type:`mstitch` has two constructors:

.. code::

   mstitch::mstitch(std::string id, mpoint prox, mpoint dist, int tag = 0)
   mstitch::mstitch(std::string id, mpoint dist, int tag = 0)

If the proximal point is omitted, it will be inferred from the point at which
the stitch is attached to its parent.

The :cpp:type:`stitch_builder` class collects the stitches with the ``add`` method:

.. code::

   stitch_builder::add(mstitch, const std::string& parent_id, double along = 1.)
   stitch_builder::add(mstitch, double along = 1.)

The first stitch will have no parent. If no parent id is specified for a subsequent
stitch, the last stitch added will be used as parent. The ``along`` parameter
must lie between zero and one inclusive, and determines the point of attachment
as a relative position between the parent's proximal and distal points.

A :cpp:type:`stitched_morphology` is constructed from a :cpp:type:`stitch_builder`,
and provides both the :cpp:type:`morphology` built from the stitches, and methods
for querying the extent of individual stitches.

.. cpp:class:: stitched_morphology

   .. cpp:function:: stitched_morphology(const stitch_builder&)
   .. cpp:function:: stitched_morphology(stitch_builder&&)

   Construct from a ``stitch_builder``. Note that constructing from an
   rvalue is more efficient, as it avoids making a copy of the underlying
   tree structure.

   .. cpp:function:: arb::morphology morphology() const

   Return the constructed morphology object.

   .. cpp:function:: region stitch(const std::string& id) const

   Return the region expression corresponding to the specified stitch.

   .. cpp:function:: std::vector<msize_t> segments(const std::string& id) const

   Return the collection of segments by index comprising the specified stitch.

   .. cpp:function:: label_dict labels(const std::string& prefix="") const

   Provide a :cpp:type:`label_dict` with a region entry for each stitch; if
   a prefix is provided, this prefix is applied to each segment id to determine
   the region labels.

Example code, constructing a cable cell from a T-shaped morphology specified
by two stitches:

.. code::

   using namespace arb;

   mpoint soma0{0, 0, 0, 10};
   mpoint soma1{20, 0, 0, 10};
   mpoint dend_end{10, 100, 0, 1};

   stitch_builder builder;
   builder.add({"soma", soma0, soma1, 1});
   builder.add({"dend", dend_end, 4}, "soma", 0.5);

   stitched_morphology stitched(std::move(builder));

   auto dec = decor{}.paint("\"soma\"", density("hh"));

   cable_cell cell(stitched.morphology(), dec, stitched.labels());


.. _locsets-and-regions:

Identifying sites and subsets of the morphology
-----------------------------------------------

.. todo::

   TODO: Region and locset documentation.


Translating regions and locsets to cables and locations
-------------------------------------------------------

.. todo::

   TODO: ``mprovider``, ``mextent`` and ``thingify``.


From morphologies to points and segments
----------------------------------------

The :cpp:type:`morphology` class has the ``branch_segments`` method for
returning a vector of :cpp:type:`msegment` objects that describe the geometry
of that branch. However, determining the position in space of an
:cpp:type:`mlocation`, for example, requires some assumptions about how to
position points which fall inside a morphological segment.

The :cpp:type:`place_pwlin` class takes a :cpp:type:`morphology` (and
optionally an :cpp:type:`isometry`) and interprets it as describing a
piecewise-linear object in space. It can then be queried to find the 3-d
positions in space of points on the morphology and the extents in space of
morphological sub-regions.

Because the morphology need not be contiguous in space, a position query can
potentially give more than one possible answer. Similarly, a description of a
cable in terms of segments or partial segments in space may include multiple
zero-length components as a result of such discontinuities.

.. cpp:class:: place_pwlin

   .. cpp:function:: place_pwlin(const morphology&, const isometry& = isometry())

      Construct a piecewise linear placement of the morphology in space,
      optionally applying the given isometry.

   .. cpp:function:: mpoint at(mlocation) const

      Return any single point corresponding to the given :cpp:class:`mlocation`
      in the placement.

   .. cpp:function:: std::vector<mpoint> all_at(mlocation) const

      Return all points corresponding to the given :cpp:class:`mlocation` in
      the placement.

   .. cpp:function:: std::vector<msegment> segments(const mextent&) const

      Return any minimal collection of segments and partial segments whose
      union is coterminous with the given :cpp:class:`mextent` in the placement.

   .. cpp:function:: std::vector<msegment> all_segments(const mextent&) const

      Return the maximal set of segments and partial segments whose
      union is coterminous with the given :cpp:class:`mextent` in the placement.

   .. cpp:function:: closest(double x, double y, double z) -> std::pair<mpoint, double>

      Find the closest location to p. Returns the location and its distance from the input coordinates.

Isometries
^^^^^^^^^^

The one cellular morphology may be used to represent multiple cable cells
which are in principle sited in different locations and orientations.
An explicit isometry allows the one morphology to be repositioned so as
to answer location queries on such cells.

An isometry consists of a rotation and a translation. Isometries can be
composed; as interpreted by Arbor, translations are always regarded as
being relative to the absolute, extrinsic co-ordinate system, while
rotations are interpreted as *intrinsic rotations*: rotations are always
applied with respect to the coordinate system carried with the object,
not the absolute co-ordinate axes.

.. cpp:class:: isometry

   .. cpp:function:: isometry()

      Construct an identity isometry.

   .. cpp:function:: static isometry translate(double x, double y, double z)

      Construct a translation (x, y, z) with respect to the extrinsic coordinate system.

   .. cpp:function:: template <typename Point> static isometry translate(const Point& p)

      Construct a translation (p.x, p.y, p.z) from an arbitrary object with the corresponding
      public member variables.

   .. cpp:function:: static isometry rotate(double theta, double x, double y, double z)

      Construct a rotation of theta radians about the axis (x, y, z) with respect to the intrinsic coordinate system.

   .. cpp:function:: template <typename Point> static isometry translate(double theta, const Point& p)

      Construct a rotation of theta radians about the (p.x, p.y, p.z) from an arbitrary object with the corresponding
      public member variables.

   .. cpp:function:: template <typename Point> Point apply(Point p) const

      The Point object is interpreted as a point in space given by public member variables x, y, and z.
      The isometry is applied to the point (x, y, z), and a copy of ``p`` is returned with the new
      coordinate values.

.. cpp:function:: isometry operator*(const isometry& a, const isometry& b)

      Compose two isometries to form a new isometry which applies the intrinsic rotation of *b*, and
      then the intrinsic rotation of *a*, together with the translations of both *a* and *b*.

.. _cv-policies:

Discretisation and CV policies
------------------------------

The set of boundary points used by the simulator is determined by a
:ref:`CV policy <morph-cv-policies>`. These are objects of type
:cpp:class:`cv_policy`, which has the following public methods:

.. cpp:class:: cv_policy

   .. cpp:function:: locset cv_boundary_points(const cable_cell&) const

   Return a locset describing the boundary points for CVs on the given cell.

   .. cpp:function:: region domain() const

   Give the subset of a cell morphology on which this policy has been declared,
   as a morphological ``region`` expression.

Specific CV policy objects are created by functions described below (strictly
speaking, these are class constructors for classes are implicit converted to
``cv_policy`` objects). These all take a ``region`` parameter that restrict the
domain of applicability of that policy; this facility is useful for specifying
differing discretisations on different parts of a cell morphology. When a CV
policy is constrained in this manner, the boundary of the domain will always
constitute part of the CV boundary point set.

CV policies can be :ref:`composed <morph-cv-composition>` with ``+`` and ``|`` operators.
For two policies
*A* and *B*, *A* + *B* is a policy which gives boundary points from both *A*
and *B*, while *A* | *B* is a policy which gives all the boundary points from
*B* together with those from *A* which do not within the domain of *B*.
The domain of *A* + *B* and *A* | *B* is the union of the domains of *A* and
*B*.

``cv_policy_single``
^^^^^^^^^^^^^^^^^^^^

.. code::

    cv_policy_single(region domain = reg::all())

Use one CV for the whole cell, or one for each connected component of the
supplied domain.

``cv_policy_explicit``
^^^^^^^^^^^^^^^^^^^^^^

.. code::

   cv_policy_explicit(locset locs, region domain = reg::all())

Use the points given by ``locs`` for CV boundaries, optionally restricted to the
supplied domain.

``cv_policy_every_segment``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

   cv_policy_every_segment(region domain = reg::all())

Use every segment in the morphology as a CV, optionally
restricted to the supplied domain. Each fork point in the domain is
represented by a trivial CV.

``cv_policy_fixed_per_branch``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    cv_policy_fixed_per_branch(unsigned cv_per_branch, region domain, cv_policy_flag::value flags = cv_policy_flag::none);

    cv_policy_fixed_per_branch(unsigned cv_per_branch, cv_policy_flag::value flags = cv_policy_flag::none):

For each branch in each connected component of the domain (or the whole cell,
if no domain is given), evenly distribute boundary points along the branch so
as to produce exactly ``cv_per_branch`` CVs.

By default, CVs will terminate at branch ends. If the flag
``cv_policy_flag::interior_forks`` is given, fork points will be included in
non-trivial, branched CVs and CVs covering terminal points in the morphology
will be half-sized.


``cv_policy_max_extent``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    cv_policy_max_extent(double max_extent, region domain, cv_policy_flag::value flags = cv_policy_flag::none);

    cv_policy_max_extent(double max_extent, cv_policy_flag::value flags = cv_policy_flag::none):

As for ``cv_policy_fixed_per_branch``, save that the number of CVs on any
given branch will be chosen to be the smallest number that ensures no
CV will have an extent on the branch longer than ``max_extent`` micrometres.

CV discretization as mcables
----------------------------

It is often useful for the user to have a detailed view of the CVs generated for a given morphology
and :ref:`cv-policy <cv-policies>`. For example, while debugging and fine-tuning model implementations,
it can be helpful to know how many CVs a cable-cell is comprised of, or how many CVs lie on a certain
region of the cell.

The following classes and functions allow the user to inspect the CVs of a cell or region.

.. cpp:class:: cell_cv_data

   Stores the discretisation data of a cable-cell in terms of CVs and the :term:`mcables <mcable>` comprising each of
   these CVs.

   .. cpp:function:: mcable_list cables(unsigned idx) const

   Returns an vector of :term:`mcable` representing the CV at a given index.

   .. cpp:function:: std::vector<unsigned> children(unsigned idx) const

    Returns a vector of the indices of the CVs representing the children of the CV at index ``idx``.

   .. cpp:function:: unsigned parent(unsigned idx) const

    Returns the index of the CV representing the parent of the CV at index ``idx``.

   .. cpp:function:: unsigned size() const

    Returns the total number of CVs on the cell.

.. cpp:function:: std::optional<cell_cv_data> cv_data(const cable_cell& cell)

   Constructs a :cpp:class:`cell_cv_data` object representing the CVs comprising the cable-cell according
   to the :cpp:class:`cv_policy` provided in the :cpp:class:`decor` of the cell. Returns ``std::nullopt_t``
   if no :cpp:class:`cv_policy` was provided in the decor.

.. cpp:class:: cv_proportion

   .. cpp:member:: unsigned idx

      Index of the CV.

   .. cpp:member:: double proportion

      Proportion of the CV by area.

.. cpp:function:: std::vector<cv_proportion> intersect_region(const region& reg, const cell_cv_data& cvs, bool integrate_by_length=false)

   Returns a vector of :cpp:class:`cv_proportion` identifying the indices of the CVs from the :cpp:class:`cell_cv_data`
   argument that lie in the provided region, and how much of each CV belongs to that region. The proportion of CV lying
   in the region is the area proportion if ``integrate_by_length`` is false, otherwise, it is the length proportion.

Supported morphology formats
============================

Arbor supports morphologies described using the SWC file format and the NeuroML file format.

.. _cppswc:

SWC
---

Arbor supports reading morphologies described using the
`SWC <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_ file format. And
has three different interpretation of that format.

A :cpp:func:`parse_swc()` function is used to parse the SWC file and generate a :cpp:type:`swc_data` object.
This object contains a vector of :cpp:type:`swc_record` objects that represent the SWC samples, with a number of
basic checks performed on them. The :cpp:type:`swc_data` object can then be used to generate a
:cpp:type:`morphology` object using one of the following functions: (See the morphology concepts
:ref:`page <morph-formats>` for more details).

  * :cpp:func:`load_swc_arbor`
  * :cpp:func:`load_swc_neuron`

.. cpp:class:: swc_record

   .. cpp:member:: int id

      ID of the record

   .. cpp:member:: int tag

       Structure identifier (tag).

   .. cpp:member:: double x

      x coordinate in space.

   .. cpp:member:: double y

      y coordinate in space.

   .. cpp:member:: double z

      z coordinate in space.

   .. cpp:member:: double r

      Sample radius.

   .. cpp:member:: int parent_id

      Record parent's sample ID.

.. cpp:class:: swc_data

   .. cpp:member:: std::string metadata

      Contains the comments of an SWC file.

   .. cpp:member:: std::vector<swc_record> records

      Stored the list of samples from an SWC file, after performing some checks.

.. cpp:function:: swc_data parse_swc(std::istream&)

   Returns an :cpp:type:`swc_data` object given an std::istream object.

.. cpp:function:: morphology load_swc_arbor(const swc_data& data)

   Returns a :cpp:type:`morphology` constructed according to Arbor's
   :ref:`SWC specifications <formatswc-arbor>`.

.. cpp:function:: morphology load_swc_neuron(const swc_data& data)

   Returns a :cpp:type:`morphology` constructed according to NEURON's
   :ref:`SWC specifications <formatswc-neuron>`.

.. _cppasc:

Neurolucida ASCII
-----------------

Arbor supports reading morphologies described using the
:ref:`Neurolucida ASCII file format <formatasc>`.

The :cpp:func:`parse_asc()` function is used to parse the SWC file and generate a :cpp:type:`asc_morphology` object:
a simple struct with two members representing the morphology and a label dictionary with labeled
regions and locations.

.. cpp:class:: asc_morphology

   .. cpp:member:: arb::morphology morphology

   .. cpp:member:: arb::label_dict labels

.. cpp:function:: asc_morphology load_asc(const std::string& filename)

   Parse a Neurolucida ASCII file.
   Throws an exception if there is an error parsing the file.


.. _cppneuroml:

NeuroML
-------

Arbor offers limited support for models described in `NeuroML version 2
<https://neuroml.org/neuromlv2>`_. All classes and functions provided by the
``arborio`` library are provided in the ``arborio`` namespace.

NeuroML2 morphology support
^^^^^^^^^^^^^^^^^^^^^^^^^^^
NeuroML documents are represented by the ``arborio::neuroml`` class,
which in turn provides methods for the identification and translation
of morphology data. ``neuroml`` objects are moveable and move-assignable,
but not copyable.

An implementation limitation restricts valid segment id values to
those which can be represented by an ``unsigned long long`` value.

``arborio::neuroml`` methods can throw an ``arborio::xml_error`` in the instance that
the underlying XML library reports a problem that cannot be handled by the ``arborio``
library. Otherwise, exceptions derived from ``aborio::neuroml_exception`` can be thrown
when encountering problems interpreting the NeuroML document (see :ref:`cppneuromlexceptions` below).

Special parsing behaviour can be invoked through the use of an enum value in the `neuroml_options`
namespace.

.. cpp:class:: neuroml

   .. cpp:function:: neuroml(std::string)

   Build a NeuroML document representation from the supplied string.

   .. cpp:function:: std::vector<std::string> cell_ids() const

   Return the id of each ``<cell>`` element defined in the NeuroML document.

   .. cpp:function:: std::vector<std::string> morphology_ids() const

   Return the id of each top-level ``<morphology>`` element defined in the NeuroML document.

   .. cpp:function:: std::optional<nml_morphology_data> morphology(const std::string&, enum neuroml_options::value = neuroml_options::none) const

   Return a representation of the top-level morphology with the supplied identifier, or
   ``std::nullopt`` if no such morphology could be found.

   .. cpp:function:: std::optional<nml_morphology_data> cell_morphology(const std::string&, enum neuroml_options::value = neuroml_options::none) const

   Return a representation of the morphology associated with the cell with the supplied identifier,
   or ``std::nullopt`` if the cell or its morphology could not be found.

.. cpp:enum:: neuroml_options::value

   .. cpp:enumerator:: none

   Perform no special parsing.

   .. cpp:enumerator:: allow_spherical_root

   Replace a zero-length root segment of constant radius with a Y-axis aligned
   cylindrical segment of the same radius and with length twice the radius. This
   cylinder will have the equivalent surface area to a sphere of the given radius.

   All child segments will connect to the centre of this cylinder, no matter the value of any ``fractionAlong`` attribute.

The morphology representation contains the corresponding Arbor ``arb::morphology`` object,
label dictionaries for regions corresponding to its segments and segment groups by name
and id, and a map providing the explicit list of segments contained within each defined
segment group.

.. cpp:class:: nml_morphology_data

   .. cpp:member:: std::optional<std::string> cell_id

   The id attribute of the cell that was used to find the morphology in the NeuroML document, if any.

   .. cpp:member:: std::string id

   The id attribute of the morphology.

   .. cpp:member:: arb::morphology morphology

   The corresponding Arbor morphology.

   .. cpp:member:: arb::label_dict segments

   A label dictionary with a region entry for each segment, keyed by the segment id (as a string).

   .. cpp:member:: arb::label_dict named_segments

   A label dictionary with a region entry for each name attribute given to one or more segments.
   The region corresponds to the union of all segments sharing the same name attribute.

   .. cpp:member:: arb::label_dict groups

   A label dictionary with a region entry for each defined segment group

   .. cpp:member:: std::unordered_map<std::string, std::vector<unsigned long long>> group_segments

   A map from each segment group id to its corresponding collection of segments.


.. _cppneuromlexceptions:

Exceptions
^^^^^^^^^^

All NeuroML-specific exceptions are defined in ``arborio/neuroml.hpp``, and are
derived from ``arborio::neuroml_exception`` which in turn is derived from ``std::runtime_error``.
With the exception of the ``nml_no_document`` exception, all contain an unsigned member ``line``
which is intended to identify the problematic construct within the document.

.. cpp:class:: nml_no_document: neuroml_exception

   A request was made to parse text which could not be interpreted as an XML document.

.. cpp:class:: nml_parse_error: neuroml_exception

   Failure parsing an element or attribute in the NeuroML document. These
   can be generated if the document does not confirm to the NeuroML2 schema,
   for example.

.. cpp:class:: nml_bad_segment: neuroml_exception

   A ``<segment>`` element has an improper ``id`` attribue, refers to a non-existent
   parent, is missing a required parent or proximal element, or otherwise is missing
   a mandatory child element or has a malformed child element.

.. cpp:class:: nml_bad_segment_group: neuroml_exception

   A ``<segmentGroup>`` element has a malformed child element or references
   a non-existent segment group or segment.

.. cpp:class:: nml_cyclic_dependency: neuroml_exception

   A segment or segment group ultimately refers to itself via ``parent``
