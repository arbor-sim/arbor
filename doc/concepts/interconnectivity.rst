.. _interconnectivity:

Interconnectivity
=================

Networks can be regarded as graphs, where the nodes are locations on cells and the edges
describe the communications between them. In Arbor, two sorts of edges are modelled: a
:term:`connection` abstracts the propagation of action potentials (:term:`spikes <spike>`) through the network,
while a :term:`gap junction` is used to describe a direct electrical connection between two locations on two cells.
Connections only capture the propagation delay and attenuation associated with spike
connectivity: the biophysical modelling of the chemical synapses themselves is the
responsibility of the target cell model.

Connection sites and gap junction sites are defined on locations on cells as part of the
:ref:`cell description <modelcelldesc>`.
A recipe lets you define which sites are connected to which.

.. _modelconnections:

.. glossary::

   connection
      Connections implement chemical synapses between **source** and **target** cells and are characterized
      by having a transmission delay.

      Connections in Arbor are defined in two steps:

      1. Create labeled **source** and **target** on two separate cells as part of their
         :ref:`cell descriptions <modelcelldesc>` in the :ref:`recipe <modelrecipe>`. Sources typically
         generate spikes. Targets are typically synapses with associated biophysical model descriptions.
         Each labeled group of sources or targets may contain multiple items on possibly multiple locations
         on the cell.
      2. Declare the connection in the recipe *on the target cell*:  from a source identified using
         a :gen:`global_label`; a target identified using a :gen:`local_label` (:gen:`gid` of target is
         the argument of the recipe method); a connection delay and a connection weight.

   spike
   action potential
      Spikes travel over :term:`connections <connection>`. In a synapse, they generate an event.

   event
      In a synapse :term:`spikes <spike>` generate events, which constitute stimulation of the synapse
      mechanism and the transmission of a signal. A synapse may receive events directly from an
      :term:`event generator`.

   event generator
      Externally stimulate a synapse. Events can be delivered on a schedule.
      See :py:class:`arbor.event_generator` for details.

.. _modelgapjunctions:

.. glossary::

   gap junction
      Gap junctions represent electrical synapses where transmission between cells is bidirectional and direct.
      They are modelled as a conductance between two **gap junction sites** on two cells.

      Similarly to `Connections`, Gap Junctions in Arbor are defined in two steps:

      1. Create labeled **gap junction sites** on two separate cells as part of their
         :ref:`cell descriptions <modelcelldesc>` in the :ref:`recipe <modelrecipe>`.
         Each labeled group of gap junctions may contain multiple items on possibly multiple locations
         on the cell.
      2. Declare the Gap Junction connections in the recipe *on the local cell*: from a peer **gap junction site**
         identified using a :gen:`global_label`; to a local **gap junction site** identified using a
         :gen:`local_label` (:gen:`gid` of the site is implicitly known); and a conductance in μS.
         Two of these connections are needed, on each of the peer and local cells.

   .. Note::
      Only cable cells support gap junctions as of now.

API
---

* :ref:`Python <pyinterconnectivity>`
* :ref:`C++ <cppinterconnectivity>`
