.. _formatneuroml:

NeuroML support
===============

Arbor offers limited support for models described in
`NeuroML version 2 <https://neuroml.org/neuromlv2>`_.
This is not built by default, but can be enabled by
providing the `-DARB_NEUROML=ON` argument to CMake at
configuration time (see :ref:`install-neuroml`). This will
build the ``arbornml`` libray and defines the corresponding
``arbor::arbornml`` CMake target.

The ``arbornml`` library uses `libxml2 <http://xmlsoft.org/>`_
for XML parsing. Applications using ``arbornml`` will need to
link against ``libxml2`` in addition, though this is performed
implicitly within CMake projects that add ``arbor::arbornml``
as a link library.

All classes and functions provided by the ``arbornml`` library
are provided in the ``arbnml`` namespace.


Libxml2 interface
-----------------

Libxml2 offers threadsafe XML parsing, but not by default. If
the application uses ``arbornml`` in an unthreaded context, or
has already explicitly initialized ``libxml2``, nothing more
needs to be done. Otherwise, the ``libxml2`` function ``xmlInitParser()``
must be called explicitly.

``arbornml`` provides a helper guard object for this purpose, defined
in ``arbornml/with_xml.hpp``:

.. cpp:namespace:: arbnml

.. cpp:class:: with_xml

   An RAII guard object that calls ``xmlInitParser()`` upon construction, and
   ``xmlCleanupParser()`` upon destruction. The constructor takes no parameters.


NeuroML 2 morphology support
----------------------------

Arbor offers limited support for models described in `NeuroML version 2 <https://neuroml.org/neuromlv2>`_.
This is not built by default (see :ref:`NeuroML support <install-neuroml>` for instructions on how
to build arbor with NeuroML).

Once support is enabled, Arbor is able to parse and check the validity of morphologies described in NeuroML files,
and present the encoded data to the user.  This is more than a simple a `segment tree`.

NeuroML can encode in the same file multiple top-level morphologies, as well as cells:

.. code:: XML

   <neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
   <morphology id="m1">
       <segment id="seg-0">
           <proximal x="1" y="1" z="1" diameter="1"/>
           <distal x="2" y="2" z="2" diameter="2"/>
       </segment>
       <segmentGroup id="group-0">
           <member segment="1"/>
       </segmentGroup>
   </morphology>
   <morphology id="m2"/>
   <cell id="c1" morphology="m1"/>
   <cell id="c2">
       <morphology id="m3"/>
   </cell>
   </neuroml>

The above NeuroML description defines 2 top-level morphologies ``m1`` and ``m2`` (empty); a cell ``c1`` that uses
morphology ``m1``; and a cell ``c2`` that uses an internally defined (empty) morphology ``m3``.

Arbor can query the cells and morphologies using their ids and return all the associated morphological data for each.
The morphological data includes the actual morphology as well as the named segments and groups of the morphology.
For example, the above ``m1`` morphology has one named segment ``seg-0`` and one named group ``group-0`` that are
both represented using Arbor's :ref:`region expressions <labels-expressions>`.

C++
^^^

NeuroML documents are represented by the ``arbnml::neuroml`` class,
which in turn provides methods for the identification and translation
of morphology data. ``neuroml`` objects are moveable and move-assignable, but not copyable.

An implementation limitation restrictes valid segment id values to
those which can be represented by an ``unsigned long long`` value.

.. cpp:class:: neuroml

   .. cpp:function:: neuroml(std::string)

   Build a NeuroML document representation from the supplied string.

   .. cpp:function:: std::vector<std::string> cell_ids() const

   Return the id of each ``<cell>`` element defined in the NeuroML document.

   .. cpp:function:: std::vector<std::string> morphology_ids() const

   Return the id of each top-level ``<morphology>`` element defined in the NeuroML document.

   .. cpp:function:: std::optional<morphology_data> morphology(const std::string&) const

   Return a representation of the top-level morphology with the supplied identifier, or
   ``std::nullopt`` if no such morphology could be found. Parse errors or an inconsistent
   representation will raise an exception derived from ``neuroml_exception``.

   .. cpp:function:: std::optional<morphology_data> cell_morphology(const std::string&) const

   Return a representation of the morphology associated with the cell with the supplied identifier,
   or ``std::nullopt`` if the cell or its morphology could not be found. Parse errors or an
   inconsistent representation will raise an exception derived from ``neuroml_exception``.

The morphology representation contains the corresponding Arbor ``arb::morphology`` object,
label dictionaries for regions corresponding to its segments and segment groups by name
and id, and a map providing the explicit list of segments contained within each defined
segment group.

.. cpp:class:: morphology_data

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

   A map from taking each segment group id to its corresponding collection of segments.


Exceptions
----------

All NeuroML-specific exceptions are defined in ``arbornml/nmlexcept.hpp``, and are
derived from ``arbnml::neuroml_exception`` which in turn is derived from ``std::runtime_error``.
With the exception of the ``no_document`` exception, all contain an unsigned member ``line``
which is intended to identify the problematic construct within the document.

.. cpp:class:: xml_error: neuroml_exception

   A generic XML error generated by the ``libxml2`` library.

.. cpp:class:: no_document: neuroml_exception

   A request was made on an :cpp:class:`neuroml` document without any content.

.. cpp:class:: parse_error: neuroml_exception

   Failure parsing an element or attribute in the NeuroML document. These
   can be generated if the document does not confirm to the NeuroML2 schema,
   for example.

.. cpp:class:: bad_segment: neuroml_exception

   A ``<segment>`` element has an improper ``id`` attribue, refers to a non-existent
   parent, is missing a required parent or proximal element, or otherwise is missing
   a mandatory child element or has a malformed child element.

.. cpp:class:: bad_segment_group: neuroml_exception

   A ``<segmentGroup>`` element has a malformed child element or references
   a non-existent segment group or segment.

.. cpp:class:: cyclic_dependency: neuroml_exception

   A segment or segment group ultimately refers to itself via ``parent``
   or ``include`` elements respectively.


