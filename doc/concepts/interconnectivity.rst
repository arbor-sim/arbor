.. _co_interconnectivity:

Interconnectivity
=================

Networks can be regarded as a sort of graph, where the nodes are cells and the edges
describe the communications between them. In Arbor, two sorts of edges are modelled: a
**connection** abstracts the propagation of action potentials (spikes) through the network,
while a **gap junction** is used to describe a direct electrical connection between two cells.
Connections only capture the propagation delay and attenuation associated with spike
connectivity: the biophysical modelling of the chemical synapses themselves is the
responsibility of the target cell model.

Connection sites and gap junction sites are defined on locations on cells (more on cells :ref:`here <modelcells>`). A recipe lets you define which sites are connected to which.

.. _modelconnections:

Connections
-----------

Connections implement chemical synapses between **source** and **target** cells and are characterized by having a transmission delay.

Connections in Arbor are defined in two steps:

1. Create **Source** and **Target** on two cells: a source defined on one cell, and a target defined on another.
2. Declare the connection in the :ref:`recipe <modelrecipe>`: with a source and target identified using :gen:`cell_member`, a connection delay and a connection weight.

.. _modelgapjunctions:

Gap junctions
-------------

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
