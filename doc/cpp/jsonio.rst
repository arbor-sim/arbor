.. _cppformats:

Json Formats
============

.. cpp:function:: cable_cell_parameter_set load_cable_cell_parameter_set(std::istream stream)

    Loads the :cpp:class:`cable_cell_parameter_set` from a ``std::istream`` of a JSON file.
    The file must follow the structure described in the
    :ref:`default parameters formats page <formatsdefault>`.
    The resulting :cpp:class:`cable_cell_parameter_set` can be used in the :cpp:class:`decor`
    of a :cpp:class:`cable_cell` or in the :cpp:class:`cable_cell_global_properties`.

.. cpp:function:: decor load_decor(std::istream stream)

    Loads the :cpp:class:`decor` from a ``std::istream`` of a JSON file. The file must follow
    the structure described in the :ref:`decor formats page <formatsdecor>`. The resulting
    :cpp:class:`decor` can be used in the creation of a :cpp:class:`cable_cell`.

.. cpp:function:: void store_cable_cell_parameter_set(const arb::cable_cell_parameter_set& set , std::ostream& stream)

    Represents the :cpp:class:`cable_cell_parameter_set` as a JSON object with the structure
    described in the :ref:`default parameters formats page <formatsdefault>` and writes it into
    ``std::ostream``.

.. cpp:function:: void store_decor(const arb::decor& decor, std::ostream& stream)

    Represents the :cpp:class:`decor` as a JSON file following the structure described
    in the :ref:`decor formats page <formatsdecor>` and writes it into ``std::ostream``.


