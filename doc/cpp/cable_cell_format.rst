.. _cppcablecellformat:

Description Format
==================

Arbor provides readers and writers for describing :ref:`label dictionaries <labels>`,
:ref:`decoration objects <cablecell-decoration>`, :ref:`morphologies <morph>` and
:ref:`cable cells <cablecell>`, referred to here as *arbor-components*.

A detailed description of the s-expression format used to describe each of these components
can be found :ref:`here <formatcablecell>`.

Reading and writing of the arbor-component description format is delegated to the ``arborio``
library and the responsible classes and functions are present in the ``arborio`` namespace.

The arbor-components and meta-data
----------------------------------

.. cpp:type:: cable_cell_variant = std::variant<arb::morphology, arb::label_dict, arb::decor, arb::cable_cell>

.. cpp:type:: template <typename T> parse_hopefully = arb::util::expected<T, cableio_parse_error>

   ``arborio::cableio_parse_error`` is derived from ``arb::arbor_exception`` which in turn is derived
   from ``std::runtime_error``. It contains information about the ``line`` and ``position`` of an encountered
   error in a document.

   ``arb::util::expected`` contains either an object of type ``T`` or an error object.

.. cpp:class:: meta_data

   .. cpp:member:: std::string version

      Stores the version of the format being used.

.. cpp:class:: cable_cell_component

   .. cpp:member:: meta_data meta

      Stores meta-data pertaining to the description of a cable cell component.

   .. cpp:member:: cable_cell_variant component

      Stores one of :cpp:class:`decor`, :cpp:class:`label_dict`, :cpp:class:`morphology` or :cpp:class:`cable_cell`.

Reading arbor-components
------------------------

.. cpp:function:: parse_hopefully<cable_cell_component> parse_component(const std::string&)

   This function will attempt to construct a :cpp:class:`cable_cell_component` object by parsing the
   contents of a string. It will return a :cpp:type:`parse_hopefully` containing the constructed object,
   or, if parsing fails, a helpful ``cableio_parse_error``.

.. cpp:function:: parse_hopefully<cable_cell_component> parse_component(std::istream&)

   Performs the same functionality as ``parse_component`` above, but starting from
   ``std::istream``.

Writing arbor-components
------------------------

.. cpp:function:: std::ostream& write_component(std::ostream&, const cable_cell_component&)

   Writes the :cpp:class:`cable_cell_component` object to the given ``std::ostream``.

.. cpp:function:: std::ostream& write_component(std::ostream& o, const arb::decor& x, const meta_data& m = {})

   Constructs a :cpp:class:`cable_cell_component` from a :cpp:class:`decor` object, and optional
   :cpp:class:`meta_data`. If no meta_data is provided, the most recent version of
   the format is used to create it. The resulting object is written to the given ``std::ostream``.

.. cpp:function:: std::ostream& write_component(std::ostream& o, const arb::label_dict& x, const meta_data& m = {})

   Constructs a :cpp:class:`cable_cell_component` from a :cpp:class:`label_dict` object, and optional
   :cpp:class:`meta_data`. If no meta_data is provided, the most recent version of
   the format is used to create it. The resulting object is written to the given ``std::ostream``.

.. cpp:function:: std::ostream& write_component(std::ostream& o, const arb::morphology& x, const meta_data& m = {})

   Constructs a :cpp:class:`cable_cell_component` from a :cpp:class:`morphology` object, and optional
   :cpp:class:`meta_data`. If no meta_data is provided, the most recent version of
   the format is used to create it. The resulting object is written to the given ``std::ostream``.

.. cpp:function:: std::ostream& write_component(std::ostream& o, const arb::cable_cell& x, const meta_data& m = {})

   Constructs a :cpp:class:`cable_cell_component` from a :cpp:class:`cable_cell` object, and optional
   :cpp:class:`meta_data`. If no meta_data is provided, the most recent version of
   the format is used to create it. The resulting object is written to the given ``std::ostream``.