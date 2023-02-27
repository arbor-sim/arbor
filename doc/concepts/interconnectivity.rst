.. _interconnectivity:

Interconnectivity
=================

Networks can be regarded as graphs, where the nodes are locations on cells and the edges
describe the communications between them. In Arbor, two sorts of edges are modelled: a
:term:`connection` abstracts the propagation of action potentials (:term:`spikes <spike>`)
through the network, while a :term:`gap junction connection` is used to describe a direct
electrical connection between two locations on two cells.
Connections only capture the propagation delay and attenuation associated with spike
connectivity: the biophysical modelling of the chemical synapses themselves is the
responsibility of the target cell model.

Connection sites and gap junction sites are defined on locations on cells as
part of the :ref:`cell description <modelcelldesc>`.
These sites as such are not connected yet, however the :ref:`recipe <modelrecipe>`
exposes a number of callbacks to form connections and gap junctions between sites.
The recipe callbacks are interrogated during simulation creation.

In addition, simulations may update their connectivity by building a new
connection table outside calls to `run`, for example

.. code-block:: python

    rec = recipe()
    dec = arb.domain_decomposition(rec, ctx)
    sim = arb.simulation(rec, ctx, dec)

    # run simulation for 0.25ms with the basic connectivity
    sim.run(0.25, 0.025)

    # extend the recipe to more connections
    rec.add_connections()
    #  use `connections_on` to build a new connection table
    sim.update_connections(rec)

    # run simulation for 0.25ms with the extended connectivity
    sim.run(0.5, 0.025)

This will completely replace the old table, previous connections to be retained
must be explicitly included in the updated callback. This can also be used to
update connection weights and delays. Note, however, that there is currently no
way to introduce new sites to the simulation, nor any changes to gap junctions.

The ``update_connections`` method accepts either a full ``recipe`` (but will
**only** use the ``connections_on`` and ``events_generators`` callbacks) or a
``connectivity``, which is a reduced recipe exposing only the relevant callbacks.
Currently ``connectivity`` is only available in C++; Python users have to pass a
full recipe.

.. warning::

   The semantics of connection updates are subtle and might produce surprising
   results if handled carelessly. In particular, spikes in-flight over a
   connection will *always* be delivered, even if the connection has been
   deleted before the time of delivery has passed (`= t_emitted +
   connection_delay`). As Arbor's connection model joins processes on the axon,
   the synaptic cleft, and the receiving synapse into a simple pair `(weight,
   delay)` it is unclear 'where' the action potential is located at the time of
   deletion relative to the locus of disconnection. Thus, it was decided to
   deliver spike events regardless. This is will not cause issues when the
   transition is slow and smooth, ie weights decays over time towards a small
   value and then the connection is removed. However, drastic and/or frequent
   changes across busy synapses might cause unexpected behaviour.

.. note::

   Arbor uses a lazily constructed network (from the ``recipe`` callbacks) for
   good reason; storing the full connectivity (for all ``gids``) in the
   ``recipe`` can lead to prohibitively large memory footprints. Keep this in
   mind when designing your connectivity and heed the consequences of doing I/O
   in these callbacks. This is doubly important when using models with dynamic
   connectivity where the temptation to store all connections is even larger and
   each call to ``update`` will re-evaluate the corresponding callbacks.

Cross-Simulator Interaction
===========================

The usual recipe can be used to declare connections to the world outside of
Arbor similar to how internal (=both source and target are Arbor's
responsibility) connections are handled.

.. code-block:: c++

    struct recipe(arb::recipe) {
      // Rest as ever before
      std::vector<arb::ext_cell_connection> external_connections_on(arb::cell_gid_type) const override {
          return {{arb::cell_remote_label_type{42,  // External GID
                                               23}, // per-gid tag
                   arb::cell_local_label_type{"tgt"},
                   weight,
                   delay}};
      }
    };

Note that Arbor now recognizes two sets of ``GID``: An external and an internal
set. This allows both Arbor and the coupled simulation to keep their own
numbering schemes. However, Arbor will tag external cells and spikes by setting
their ``GID``s' most significant bit. This _halves_ the effecively available
``GID``s.

To consume external spike events, a specialised ``context`` must be created by
calling

.. code-block:: c++

    auto ctx = arb::make_context({}, local, inter);

where ``local`` is an MPI intracommunicator and ``inter`` an MPI
intercommunicator. ``inter`` is required to bridge the Arbor (``local``) and
external simulator's respective MPI communicators. Note, that the exchange
protocol _requires_ the semantics of an intercommunicator, passing anything else
will result in an exception. You can create an intercommunicator in two main
ways. First by splitting a pre-existing intercommunicator using
``MPI_Comm_split(4)`` and then calling ``MPI_Intercomm_create(7)`` on the
result. This approach produces a single binary that goes down two different
route, one calling Arbor and the other coupled simulation. Our ``remote``
example works this way. Second, using ``MPI_Comm_connect(5)`` and
``MPI_Comm_accept(5)`` will result in two completely separate binaries that can
communicate over the generated intercommunicator. Please consult the MPI
documentation for more details on these methods.

Data Plane and Spike Exchange
-----------------------------

The actual communication is performed in two steps, one to collect the number
spikes from each participating task via ``MPI_Allgather(7)`` and the second to
transfer the actual payload by ``MPI_Allgatherv(8)``. Note that over an
intercommunicator, allgather will work slightly unintuitively by concatenating
all results of a given 'side' of the intercommunicator and broadcasting that to
the other 'side' and vice-versa. For example, assume Arbor has three MPI tasks,
sending ``a0``, ``a1``, and ``a2`` respectively and the coupled package has two
sending ``b0`` and ``b1``. After allgather, each rank of the three ranks of
Arbor will have ``[b0, b1]`` and the other two ranks will have ``[a0, a1, a2]``.
We package this in the suplemental header ``arbor/communication/remote.hpp`` as
``gather_all``.

Please refer to our developer's documentation for the actual spike exchange
process.

Control Plane and Epochs
------------------------

Before initiating the actual simulation, Arbor sets the ``epoch`` length to half
the minimal delay in the global network. The minimal delay can be queried using
``simulation::min_delay``.

Before the start of each ``epoch``, a control message must be exchanged between
the root -- ie rank 0 -- process of the Arbor process and that of coupled
simulation. The control message is transferred by ``MPI_Sendrecv(12)`` of a byte
buffer of length 1024. The payload comprises
1. A single byte magic number
2. A three byte version number
3. A single byte message tag
4. A binary representation of a C ``struct`` message

All constants and types are defined in ``arbor/communication/remote.hpp``;
currently Arbor understands and utilises the following message types:

If ``abort`` is received or sent Arbor will shut down at the next possible
moment without performing any further work and potentially terminating all
outstanding communication. An exception will be raised. Note that Arbor might
terminate even without sending or receiving an ``abort`` message in exceptional
circumstances.

On ``epoch`` Arbor will commence the next epoch. Note that Arbor may expect the
last epoch to be shortened, ie when the total runtime is not a multiple of the
epoch length.

``Done`` signals the sending side is finished with the current simulation
period. *May* cause the receiving side to quit.

``Null`` does nothing, but reserved for future use, will currently not be sent
by Arbor.

**Important** This is a synchronous protocol which means an unannounced
 termination of either side of the coupled simulators can lead to the other
 getting stuck on a blocking call to MPI. This unlikely to cause issues in
 scenarios where both sides are launched as a single job (eg via ``SLURM``), but
 might do so where unrelated jobs are used.

Tying It All Together
---------------------

We recommend to make use of the facilities offered in
``arbor/communication/remote.hpp``, as does Arbor internally. Refer to the
``remote.cpp`` example on how they are used.

Terms and Definitions
=====================

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

         .. code-block:: python

             def connections_on(self, gid):
                 if gid + 1 < self.num_cells():
                     return [arbor.connection((gid + 1, "spike-source"), "synapse", weight, delay)]
                 else:
                     return []

   spike
   action potential
      Spikes travel over :term:`connections <connection>`. In a synapse, they generate an event.

   threshold detector
      :ref:`Placed <cablecell-place>` on a cell. Possible source of a connection.
      Detects crossing of a fixed threshold and generates corresponding events.
      Also used to record spikes for analysis. See :ref:`here
      <cablecell-threshold-detectors>` for more information.

   spike source cell
      Artificial cell to generate spikes on a given schedule, see :ref:`spike cell <spikecell>`.

   recording
      By default, spikes are used for communication, but not stored for analysis,
      however, :ref:`simulation <modelsimulation>` objects can be instructed to record spikes.

   event
      In a synapse :term:`spikes <spike>` generate events, which constitute stimulation of the synapse
      mechanism and the transmission of a signal. A synapse may receive events directly from an
      :term:`event generator`.

   event generator
      Externally stimulate a synapse. Events can be delivered on a schedule.
      See :py:class:`arbor.event_generator` for details.

.. _modelgapjunctions:

.. glossary::

   gap junction connection
      Gap junctions represent electrical synapses where transmission between cells is bidirectional and direct.
      They are modelled as a conductance between two **gap junction sites** on two cells.

      Similarly to `Connections`, Gap Junctions in Arbor are defined in two steps:

      1. Create labeled **gap junction sites** on two separate cells as part of
         their :ref:`cell descriptions <modelcelldesc>` in the :ref:`recipe
         <modelrecipe>`. Each labeled group of gap junctions may contain multiple
         items on possibly multiple locations on the cell.
      2. Declare the Gap Junction connections in the recipe *on the local cell*:
         from a peer **gap junction site** identified using a
         :gen:`global_label`; to a local **gap junction site** identified using
         a :gen:`local_label` (:gen:`gid` of the site is implicitly known); and
         a unit-less connection weight. Two of these connections are needed, on
         each of the peer and local cells. The callback `gap_junctions_on`
         returns a list of these items, eg

         .. code-block:: python

             def gap_junctions_on(self, gid):
                 n = self.num_cells
                 if gid + 1 < n and gid > 0:
                     return [arbor.gap_junction_connection((gid + 1, "gj"), "gj", weight),
                             arbor.gap_junction_connection((gid - 1, "gj"), "gj", weight),]
                 elif gid + 1 < n:
                     return [arbor.gap_junction_connection((gid + 1, "gj"), "gj", weight),]
                 if gid > 0:
                     return [arbor.gap_junction_connection((gid - 1, "gj"), "gj", weight),]
                 else:
                     return []

         Note that gap junction connections are symmetrical and thus the above
         example generates two connections, one incoming and one outgoing.

   .. Note::
      Only cable cells support gap junctions as of now.

API
---

* :ref:`Python <pyinterconnectivity>`
* :ref:`C++ <cppinterconnectivity>`
