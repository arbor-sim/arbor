.. _cablecell:

.. py:module:: arbor

Cable Cells
===========

An Arbor *cable cell* is a full description of a cell with morphology and cell
dynamics, where cell dynamics include ion species and their properties, ion
channels, synapses, gap junction sites, stimulii and spike detectors.
Arbor cable cells are constructed from a morphology and a label dictionary,
and provide a rich interface for specifying the cell's dynamics.

.. note::
    Before reading this page, it is recommended that you first read about
    :ref:`morphology descriptions <morph-morphology>`, and also
    :ref:`label dictionary <labels-dictionary>` that are used to describe
    :ref:`locations <labels-locset>` and :ref:`regions <labels-region>` on a cell.

Decoration
------------

A cell is *decorated* by specifying the distribution and placement of dynamics
on the cell, to produce a *cable cell*: a full description
of a cell morphology and its dynamics with all information required to build
a a standalone single-cell model, or as part of a larger network.

Decoration uses region and locset descriptions to specify the dynamics, and
their respective use for this purpose are reflected in the two broad classes
of *dynamics* in Arbor:

* *Painted dynamics* are applied to regions of a cell, and are associated with
  an area of the membrane or volume of the cable. Examples include
  membrane capacitance, the density distribution of an ion channel, and the
  initial concentration of an ion species.
* *Placed dynamics* are applied to locations on the cell, and are associated
  with entities that can be counted. Examples include synapses, stimulii,
  spike detectors, and gap junction sites.

Painted Dynamics
^^^^^^^^^^^^^^^^

Painted dynamics are present on regions of cells.
Some dynamics, such as membrane capacitance and the initial concentration of ion species
must be defined for all compartments. Others need only be applied where they are
present, for example ion channels.

Arbor uses a hierarchical approach to resolve the these parameters and properties.
This approach allows us to, for example, define a global default value for calcium
concentration, then provide a different values on specific cell regions.

.. _cable-painted-resolution:

.. csv-table:: Painted property resolution options
   :widths: 20, 10, 10, 10

                  ,       **region**, **cell**, **global**
   cable properties,       ✓, ✓, ✓
   ion initial conditions, ✓, ✓, ✓
   density mechnism,       ✓, --, --
   ion rev pot mechanism,  --, ✓, ✓
   ion valence,            --, --, ✓




.. note::
    When a simulation object is constructed, the backend *discretizes* the morphology
    into *compartments*. The decoration is queried to determine the cable properties of
    each compartmet, which density mechanisms, ion species, synapses and so on are
    present on each compartment.

    Some properties, such as membrane capacitance and the concentration of ion species
    must be defined for all compartments. Others need only be applied where they are
    present, for example ion channels and synapse instances.

    Arbor uses a hierarchical approach to resolve the these parameters and properties.
    This approach allows us to, for example, define a global default value for calcium
    concentration, then provide a different values on specific cell regions.

