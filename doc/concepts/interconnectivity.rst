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

Connection sites and gap junction sites are defined on locations on cells as part of the
:ref:`cell description <modelcelldesc>`.
A recipe lets you define which sites are connected to which.

.. _modelconnections:

Connections
-----------

Connections implement chemical synapses between **source** and **target** cells and are characterized
by having a transmission delay.

Connections in Arbor are defined in two steps:

1. Create **source** and **target** on two separate cells as part of their
   :ref:`cell descriptions <modelcelldesc>` in the :ref:`recipe <modelrecipe>`. Sources typically
   generate spiking events. Targets are typically synapses with associated biophysical model descriptions.
2. Declare the connection in the recipe: with the source and target identified using :gen:`cell_member`,
   a connection delay and a connection weight. The connection should be declared on the target cell.

.. _modelgapjunctions:

Gap junctions
-------------

Gap junctions represent electrical synapses where transmission between cells is bidirectional and direct.
They are modeled as a conductance between two **gap junction sites** on two cells.

Similarly to `Connections`, Gap Junctions in Arbor are defined in two steps:

1. Create a **gap junction site** on two separate cells as part of their
   :ref:`cell descriptions <modelcelldesc>` in the :ref:`recipe <modelrecipe>`.
2. Declare the Gap Junction in the recipe: with two **gap junction sites** identified using :gen:`cell_member`
   and a conductance in μS.

   .. Note::
      Only cable cells support gap junctions as of now.

API
---

* :ref:`Python <pyinterconnectivity>`
* :ref:`C++ <cppinterconnectivity>`
