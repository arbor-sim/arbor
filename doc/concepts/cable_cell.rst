.. _cablecell:

Cable cells
===========

An Arbor *cable cell* is a full :ref:`description <modelcelldesc>` of a cell
with morphology and cell dynamics like ion species and their properties, ion
channels, synapses, gap junction mechanisms, stimuli and threshold detectors.

Cable cells are constructed from three components:

* :ref:`Morphology <morph>`: a description of the geometry and branching structure of the cell shape.
* :ref:`Label dictionary <labels>`: a set of definitions and a :abbr:`DSL (domain specific language)` that refer to regions and locations on the cell morphology.
* :ref:`Decor <cablecell-decoration>`: a description of the dynamics on the cell, placed according to the named rules in the dictionary. It can reference :ref:`mechanisms` from mechanism catalogues.

When a cable cell is constructed the following steps are performed using the inputs:

1. Concrete regions and locsets are generated for the morphology for each labelled region and locset in the dictionary
2. The default values for parameters specified in the decor, such as ion species concentration, are instantiated.
3. Dynamics (mechanisms, parameters, synapses, gap junctions etc.) are instantiated on the regions and locsets as specified by the decor.

Once constructed, the cable cell can be queried for specific information about the cell, but it can't be modified (it is *immutable*).

.. Note::

    The inputs used to construct the cell (morphology, label definitions and decor) are orthogonal,
    which allows a broad range of individual cells to be constructed from a handful of simple rules
    encoded in the inputs.
    For example, take a model with the following:

    * three cell types: pyramidal, purkinje and granule.
    * two different morphologies for each cell type (a total of 6 morphologies).
    * all cells have the same basic region definitions: soma, axon, dendrites.
    * all cells of the same type (e.g. Purkinje) have the same dynamics defined on their respective regions.

    The basic building blocks required to construct all of the cells for the model would be:

    * 6 morphologies (2 for each of purkinje, granule and pyramidal).
    * 3 decors (1 for each of purkinje, granule and pyramidal).
    * 1 label dictionary that defines the region types.

.. toctree::
   :maxdepth: 2

   morphology
   labels
   mechanisms
   decor

API
---

* :ref:`Python <pycablecell>`
* :ref:`C++ <cppcablecell>`

