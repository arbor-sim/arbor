.. _cppjsonformats:

Json Formats
============

.. cpp:function:: std::variant<arb::decor, arb::cable_cell_parameter_set> load_json(const nlohmann::json&);

    Loads a ``std::variant`` of a :cpp:class:`cable_cell_parameter_set` or :cpp:class:`decor`
    from a ``nlohmann::json`` object.
    The JSON object must follow the structure described in
    :ref:`the default parameters formats page <formatsdefault>` to return a
    :cpp:class:`cable_cell_parameter_set`; or it must follow the structure described in
    :ref:`the decor formats page <formatsdecor>` to return a :cpp:class:`decor`.

.. cpp:function:: nlohmann::json write_json(const arb::cable_cell_parameter_set&)

    Returns a ``nlohmann::json`` object representing the :cpp:class:`cable_cell_parameter_set`
    according to the structure described in the :ref:`default parameters formats page <formatsdefault>`.

.. cpp:function::  nlohmann::json write_json(const arb::decor&)

    Returns a ``nlohmann::json`` object representing the :cpp:class:`decor`
    according to the structure described in the :ref:`decor formats page <formatsdecor>`.

