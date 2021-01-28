.. _pyjsonformats:

Json Formats
============

.. currentmodule:: arbor

.. py:function:: load_json(filename)

    Loads either a :class:`cable_parameter_set` or a :class:`decor` from a JSON file.

    The file must follow the structure described in the
    :ref:`default parameters formats page <formatsdefault>` to return a
    :class:`cable_parameter_set`. The returned object can then be applied to a
    :class:`single_cell_model` or the :class:`cable_global_properties` returned by
    :meth:`recipe.global_properties`. The parameters will be applied to all cells in
    the model, unless overridden at the cell or region level using the :class:`decor`.

    The file must follow the structure described in the
    :ref:`decor formats page <formatsdecor>` to return a :class:`decor`. The returned
    object can then be used in the creation of a :class:`cable_cell`.

    :param str filename: the name of the JSON file.
    :rtype: :class:`cable_parameter_set` or :class:`decor`

.. py:function:: store_json(params, filename)

    Stores the `params` representing a :class:`cable_parameter_set` to a JSON
    file following the structure described in the
    :ref:`default parameters formats page <formatsdefault>`.

    :param cable_parameter_set params: the set of params to be stored.
    :param str filename: the name of the JSON file.

.. py:function:: store_json(decor, filename)

    Stores the :class:`decor` to a JSON file following the structure described
    in the :ref:`decor formats page <formatsdecor>`.

    :param decor params: the decor to be stored.
    :param str filename: the name of the JSON file.
