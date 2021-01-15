.. _pyjsonformats:

Json Formats
============

.. currentmodule:: arbor

.. py:function:: load_default_parameters(filename)

    Loads the :class:`cable_parameter_set` from a JSON file. The file must follow the
    structure described in the :ref:`default parameters formats page <formatsdefault>`.
    The resulting :class:`cable_parameter_set` can be applied to a
    :class:`single_cell_model` or the :class:`cable_global_properties` returned by
    :meth:`recipe.global_properties`.
    The parameters will be applied to all cells in the model, unless overridden
    at the cell or region level using the :class:`decor`.

    :param str filename: the name of the JSON file.
    :rtype: :class:`cable_parameter_set`

.. py:function:: load_decor(filename)

    Loads the :class:`decor` from a JSON file. The file must follow the structure
    described in the :ref:`decor formats page <formatsdecor>`. The resulting :class:`decor`
    can be used in the creation of a :class:`cable_cell`.

    :param str filename: the name of the JSON file.
    :rtype: :class:`decor`

.. py:function:: store_default_parameters(params, filename)

    Stores the `params` representing a :class:`cable_parameter_set` to a JSON
    file following the structure described in the
    :ref:`default parameters formats page <formatsdefault>`.

    :param cable_parameter_set params: the set of params to be stored.
    :param str filename: the name of the JSON file.

.. py:function:: store_decor(decor, filename)

    Stores the :class:`decor` to a JSON file following the structure described
    in the :ref:`decor formats page <formatsdecor>`.

    :param decor params: the decor to be stored.
    :param str filename: the name of the JSON file.
