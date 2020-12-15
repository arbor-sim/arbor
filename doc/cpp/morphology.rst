.. _cppmorphology:

Cable cell morphologies
=======================

Cell morphologies are required to describe a :ref:`cppcablecell`.
Morphologies can be constructed directly, or read from a number of
file formats; see :ref:`cppcablecell-morphology-construction` for details.

Morphology API
--------------

.. todo::

   TODO: Describe morphology methods.

.. _cppcablecell-morphology-construction:

Constructing cell morphologies
------------------------------

.. todo::

   TODO: Description of segment trees.


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
   cable_cell cell(stitched.morphology(), stitched.labels());

   cell.paint("\"soma\"", "hh");


Supported morphology formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Arbor supports morphologies described using the SWC file format and the NeuroML file format.

SWC
"""

Arbor supports reading morphologies described using the
`SWC <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_ file format. And
has three different interpretation of that format.

A :cpp:func:`parse_swc()` function is used to parse the SWC file and generate a :cpp:type:`swc_data` object.
This object contains a vector of :cpp:type:`swc_record` objects that represent the SWC samples, with a number of
basic checks performed on them. The :cpp:type:`swc_data` object can then be used to generate a
:cpp:type:`morphology` object using one of the following functions: (See the morphology concepts
:ref:`page <morph-formats>` for more details).

  * :cpp:func:`load_swc_arbor`
  * :cpp:func:`load_swc_allen`
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

   Returns a :cpp:type:`morphology` constructed according to Arbor's SWC specifications.

.. cpp:function:: morphology load_swc_allen(const swc_data& data, bool no_gaps=false)

   Returns a :cpp:type:`morphology` constructed according to the Allen Institute's SWC
   specifications. By default, gaps in the morphology are allowed, this can be toggled
   using the ``no_gaps`` argument.

.. cpp:function:: morphology load_swc_neuron(const swc_data& data)

   Returns a :cpp:type:`morphology` constructed according to NEURON's SWC specifications.

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


