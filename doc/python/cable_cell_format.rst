.. _pycablecellformat:

Description Format
==================

Arbor provides readers and writers for describing :ref:`label dictionaries <labels>`,
:ref:`decoration objects <cablecell-decoration>`, :ref:`morphologies <morph>` and
:ref:`cable cells <cablecell>`, referred to here as *arbor-components*.

A detailed description of the s-expression format used to describe each of these components
can be found :ref:`here <formatcablecell>`.

The arbor-components and meta-data
----------------------------------
.. currentmodule:: arbor

.. py:class:: meta_data

   .. py:attribute:: string version

      Stores the version of the format being used.

.. py:class:: cable_cell_component

   .. py:attribute:: meta_data meta

      Stores meta-data pertaining to the description of a cable cell component.

   .. py:attribute:: component

      Stores one of :class:`decor`, :class:`label_dict`, :class:`morphology` or :class:`cable_cell`.

Reading and writing arbor-components
------------------------------------

.. py:function:: load_component(filename)

   Load :class:`cable_cell_component` (decor, morphology, label_dict, cable_cell) from file.

   :param str filename: the name of the file containing the component description.
   :rtype: :class:`cable_cell_component`

.. py:function:: write_component(comp, filename)

   Write the :class:`cable_cell_component` to file.

   :param cable_cell_component comp: the component to be written to file.
   :param str filename: the name of the file.

.. py:function:: write_component(dec, filename)
   :noindex:

   Write the :class:`decor` to file. Use the most recent version of the cable cell format to construct the meta-data.

   :param decor dec: the decor to be written to file.
   :param str filename: the name of the file.

.. py:function:: write_component(dict, filename)
   :noindex:

   Write the :class:`label_dict` to file. Use the most recent version of the cable cell format to construct the meta-data.

   :param label_dict dict: the label dictionary to be written to file.
   :param str filename: the name of the file.

.. py:function:: write_component(morpho, filename)
   :noindex:

   Write the :class:`morphology` to file. Use the most recent version of the cable cell format to construct the meta-data.

   :param morphology morpho: the morphology to be written to file.
   :param str filename: the name of the file.

.. py:function:: write_component(cell, filename)
   :noindex:

   Write the :class:`cable_cell` to file. Use the most recent version of the cable cell format to construct the meta-data.

   :param cable_cell cell: the cable_cell to be written to file.
   :param str filename: the name of the file.