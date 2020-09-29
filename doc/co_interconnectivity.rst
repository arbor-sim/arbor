.. _co_interconnectivity:

Interconnectivity
#################

Networks consist of two components, nodes (cells) and edges (synapses). Arbor models the two kinds of synapses: chemical and electrical. In Arbor, a chemical synapse is called a *connection*, and an electrical synapse a *gap junction*. On cells (more on cells :ref:`here <modelcells>`) you can make connection sites and junction sites, and in a recipe you define which sites on which cells are in fact connected.

.. _modelconnections:

Connections
===========

Connections implement chemical synapses between **source** and **target** cells and are characterized by having a transmission delay.

Connections in Arbor are defined in two steps:

1. Create **Source** and **Target** on two cells: a source defined on one cell, and a target defined on another.
2. Declare the connection in the :ref:`recipe <modelrecipe>`: with a source and target idenfied using :gen:`cell_member`, a connection delay and a connection weight.

.. _modelgapjunctions:

Gap Junctions
=============

Gap junctions represent electrical synapses where transmission between cells is bidirectional and direct.
They are modeled as a conductance between two **gap junction sites** on two cells.

Similarly to `Connections`, Gap Junctions in Arbor are defined in two steps:

1. A **gap junction site** is created on each of the two cells.
   These locations need to be declared on the :ref:`cell <modelcells>`.
2. Gap Junction instantiation in the :ref:`recipe <modelrecipe>`: The **gap junction sites** are indexed using :gen:`cell_member`
   because a single cell may have more than one gap junction site.
   A gap junction is instantiated by providing two **gap junction sites'** and a conductance in μS.

   .. Note::
      Only cable cells support gap junctions as of now.

API
---

* :ref:`Python <pyinterconnectivity>`
* :ref:`C++ <cppinterconnectivity>`
