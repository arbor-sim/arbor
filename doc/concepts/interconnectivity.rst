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

.. _interconnectivity-high-level:

High Level Network Description
------------------------------

As an alternative to providing a list of connections for each cell in the :ref:`recipe <modelrecipe>`, arbor supports high-level description of a cell network. It is based around a ``network_selection`` type, that represents a selection from the set of all possible connections between cells. A selection can be created based on different criteria, such as source or destination label, cell indices and also distance between source and destination. Selections can then be combined with other selections through set algebra like expressions. For distance calculations, the location of each connection point on the cell is resolved through the morphology combined with a cell isometry, which describes translation and rotation of the cell.
Each connection also requires a weight and delay value. For this purpose, a ``network_value`` type is available, that allows to mathematically describe the value calculation using common math functions, as well random distributions.

The following example shows the relevant recipe functions, where cells are connected into a ring with additional random connections between them:

.. code-block:: python

    def network_description(self):
        seed = 42

        # create a chain
        chain = f"(chain (gid-range 0 {self.ncells}))"
        # connect front and back of chain to form ring
        ring = f"(join {chain} (intersect (source-cell {self.ncells - 1}) (destination-cell 0)))"

        # Create random connections with probability inversely proportional to the distance within a
        # radius
        max_dist = 400.0  # μm
        probability = f"(div (sub {max_dist} (distance)) {max_dist})"
        rand = f"(intersect (random {seed} {probability}) (distance-lt {max_dist}))"

        # combine ring with random selection
        s = f"(join {ring} {rand})"
        # restrict to inter-cell connections and certain source / destination labels
        s = f'(intersect {s} (inter-cell) (source-label "detector") (destination-label "syn"))'

        # fixed weight for connections in ring
        w_ring = "(scalar 0.01)"
        # random normal distributed weight with mean 0.02 μS, standard deviation 0.01 μS
        # and truncated to [0.005, 0.035]
        w_rand = f"(truncated-normal-distribution {seed} 0.02 0.01 0.005 0.035)"

        # combine into single weight expression
        w = f"(if-else {ring} {w_ring} {w_rand})"

        # fixed delay
        d = "(scalar 5.0)"  # ms delay

        return arbor.network_description(s, w, d, {})

    def cell_isometry(self, gid):
        # place cells with equal distance on a circle
        radius = 500.0 # μm
        angle = 2.0 * math.pi * gid / self.ncells
        return arbor.isometry.translate(radius * math.cos(angle), radius * math.sin(angle), 0)


The export function ``generate_network_connections`` allows the inspection of generated connections. The exported connections include the cell index, local label and location of both source and destination.


.. note::

   Expressions using distance require a cell isometry to resolve the global location of connection points.

.. note::

   A high-level description may be used together with providing explicit connection lists for each cell, but it is up to the user to avoid multiple connections between the same source and destination.

.. warning::

   Generating connections always involves additional work and may increase the time spent in the simulation initialization phase.


.. _interconnectivity-selection-expressions:

Network Selection Expressions
-----------------------------

.. label:: (gid-range begin:integer end:integer)

    A range expression, representing a range of indices in the half-open interval [begin, end).

.. label:: (gid-range begin:integer end:integer step:integer)

    A range expression, representing a range of indices in the half-open interval [begin, end) with a given step size. Step size must be positive.

.. label:: (cable-cell)

    Cell kind expression for cable cells.

.. label:: (lif-cell)

    Cell kind expression for lif cells.

.. label:: (benchmark-cell)

    Cell kind expression for benchmark cells.

.. label:: (spike-source-cell)

    Cell kind expression for spike source cells.

.. label:: (all)

    A selection of all possible connections.

.. label:: (none)

    A selection representing the empty set of possible connections.

.. label:: (inter-cell)

    A selection of all connections that connect two different cells.

.. label:: (network-selection name:string)

    A named selection within the network dictionary.

.. label:: (intersect network-selection network-selection [...network-selection])

    The intersection of at least two selections.

.. label:: (join network-selection network-selection [...network-selection])

    The union of at least two selections.

.. label:: (symmetric-difference network-selection network-selection [...network-selection])

    The symmetric difference of at least two selections.

.. label:: (difference network-selection network-selection)

    The difference of two selections.

.. label:: (difference network-selection)

    The complement or opposite of the given selection.

.. label:: (source-cell-kind kind:cell-kind)

    All connections, where the source cell is of the given type.

.. label:: (destination-cell-kind kind:cell-kind)

    All connections, where the destination cell is of the given type.

.. label:: (source-label label:string)

    All connections, where the source label matches the given label.

.. label:: (destination-label label:string)

    All connections, where the destination label matches the given label.

.. label:: (source-cell integer [...integer])

    All connections, where the source cell index matches one of the given integer values.

.. label:: (source-cell range:gid-range)

    All connections, where the source cell index is contained in the given gid-range.

.. label:: (destination-cell integer [...integer])

    All connections, where the destination cell index matches one of the given integer values.

.. label:: (destination-cell range:gid-range)

    All connections, where the destination cell index is contained in the given gid-range.

.. label:: (chain integer [...integer])

    A chain of connections between cells in the given order of in the list, such that entry "i" is the source and entry "i+1" the destination.

.. label:: (chain range:gid-range)

    A chain of connections between cells in the given order of the gid-range, such that entry "i" is the source and entry "i+1" the destination.

.. label:: (chain-reverse range:gid-range)

    A chain of connections between cells in reverse of the given order of the gid-range, such that entry "i+1" is the source and entry "i" the destination.

.. label:: (random p:real)

    A random selection of connections, where each connection is selected with the given probability.

.. label:: (random p:network-value)

    A random selection of connections, where each connection is selected with the given probability expression.

.. label:: (distance-lt dist:real)

    All connections, where the distance between source and destination is less than the given value in micro meter.

.. label:: (distance-gt dist:real)

    All connections, where the distance between source and destination is greater than the given value in micro meter.


.. _interconnectivity-value-expressions:

Network Value Expressions
-------------------------

.. label:: (scalar value:real)

    A scalar of given value.

.. label:: (network-value name:string)

    A named network value in the network dictionary.

.. label:: (distance)

    The distance between source and destination.

.. label:: (distance value:real)

    The distance between source and destination scaled by the given value.

.. label:: (uniform-distribution seed:integer begin:real end:real)

    Uniform random distribution within the interval [begin, end).

.. label:: (normal-distribution seed:integer mean:real std_deviation:real)

    Normal random distribution with given mean and standard deviation.

.. label:: (truncated-normal-distribution seed:integer mean:real std_deviation:real begin:real end:real)

    Truncated normal random distribution with given mean and standard deviation within the interval [begin, end).

.. label:: (if-else sel:network-selection true_value:network-value false_value:network-value)

    Truncated normal random distribution with given mean and standard deviation within the interval [begin, end).

.. label:: (add (network-value | real) (network-value | real) [... (network-value | real)])

    Addition of at least two network values or real numbers.

.. label:: (sub (network-value | real) (network-value | real) [... (network-value | real)])

    Subtraction of at least two network values or real numbers.

.. label:: (mul (network-value | real) (network-value | real) [... (network-value | real)])

    Multiplication of at least two network values or real numbers.

.. label:: (div (network-value | real) (network-value | real) [... (network-value | real)])

    Division of at least two network values or real numbers.
    The expression is evaluated from the left to right, dividing the first element by each divisor in turn.

.. label:: (min (network-value | real) (network-value | real) [... (network-value | real)])

    Minimum of at least two network values or real numbers.

.. label:: (max (network-value | real) (network-value | real) [... (network-value | real)])

    Maximum of at least two network values or real numbers.

.. label:: (log (network-value | real))

    Logarithm of a network value or real number.

.. label:: (exp (network-value | real))

    Exponential function of a network value or real number.



.. _interconnectivity-mut:

Mutability
----------

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
    #  use updated recipe to build a new connection table
    sim.update(rec)

    # run simulation for 0.25ms with the extended connectivity
    sim.run(0.5, 0.025)

This will completely replace the old table, previous connections to be retained
must be explicitly included in the updated callback. This can also be used to
update connection weights and delays. Note, however, that there is currently no
way to introduce new sites to the simulation, nor any changes to gap junctions.

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
   Alternatively, connections can be generated by Arbor using the network DSL 
   through the ``network_description`` callback function.

.. _interconnectivitycross:

Cross-Simulator Interaction
---------------------------

This section describes how external simulators communicating via spikes can be
connected to Arbor. For other methods of communication, translation to spikes,
e.g. from neural mass models, is needed. For coupling to microscopic
simulations, e.g. of individual ion channels, a different API is required. The
mechanism ABI might be a good fit there.

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

similarly

.. code-block:: python

    class recipe(arb.recipe):
        # Rest as ever before
        def external_connections_on(self, gid):
            return [arb.connection((42,      # external GID
                                    32),     # tag
                                    "tgt",
                                    weight,
                                    delay)]]

Note that Arbor now recognizes two sets of ``GID``\: An external and an internal
set. This allows both Arbor and the coupled simulation to keep their own
numbering schemes. However, internally Arbor will tag external cells and spikes
by setting their ``GID``\s'  most significant bit. This _halves_ the effecively
available ``GID``\s.

To consume external spike events, a specialised ``context`` must be created by
calling

.. code-block:: c++

    auto ctx = arb::make_context({}, local, inter);

or similarly in Python

.. code-block:: python

    ctx = arb.make_context(mpi=local, inter=inter)

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The actual communication is performed in two steps, one to collect the number
spikes from each participating task via ``MPI_Allgather(7)`` and the second to
transfer the actual payload by ``MPI_Allgatherv(8)``. Note that over an
intercommunicator, allgather will work slightly unintuitively by concatenating
all results of a given 'side' of the intercommunicator and broadcasting that to
the other 'side' and vice-versa. For example, assume Arbor has three MPI tasks,
sending ``a0``, ``a1``, and ``a2`` respectively and the coupled package has two
tasks, sending ``b0`` and ``b1``. After allgather, each of the three Arbor ranks
will have ``[b0, b1]`` and the two ranks of the other side will have ``[a0, a1,
a2]`` each. We package this in the suplemental header
``arbor/communication/remote.hpp`` as ``gather_spikes``. This function will
accept a ``std::vector<arb_spike>`` where ``arb_spike`` is a binary compatible
version of Arbor's internal spike type that is to be sent from the local rank of
the coupled packaged, eg ``b1`` from above. After the operation Arbor has
received the concatenation of all such vectors and the routine will return the
concatenation of all spikes produced and exported by Arbor on all ranks of the
participating package.

Please refer to our developer's documentation for more details the actual spike
exchange process. Due to the way MPI defines intercommunicators, the exchange is
the same as with intracommunicators.

Control Plane and Epochs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before initiating the actual simulation, Arbor sets the ``epoch`` length to half
the minimal delay in the global network. The minimal delay can be queried using
``simulation::min_delay`` and the epoch length is given by
``simulation::max_epoch_length``. The final epoch is optionally shorter, if the
call to ``simulation::run(T, dt)`` is given a value for ``T`` that is not an
integer multiple of the epoch length.

Before the start of each ``epoch``, a control message must be exchanged between
Arbor and the coupled simulation. The control message is transferred by use
``MPI_Allreduce(6)`` with operation ``MPI_SUMM`` on a byte buffer of length
``ARB_REMOTE_MESSAGE_LENGTH``. All processes begin with a buffer of zeroes, the
process with ``rank`` equal to ``ARB_REMOTE_ROOT`` on both sides of the
intercommunicator writes a payload comprising

1. A single byte magic number
2. A three byte version number
3. A single byte message tag
4. A binary representation of a C ``struct`` message

to its buffer. Then, the exhange is performed. This peculiar protocol yields a
simultaneous exchange in both directions across the intercommunicator without
taking order into consideration.

All constants and types -- including the messages -- are defined in
``arbor/communication/remote.hpp``; currently Arbor understands and utilises the
following message types:

If ``abort`` is received or sent Arbor will shut down at the next possible
moment without performing any further work and potentially terminating all
outstanding communication. An exception will be raised. Note that Arbor might
terminate even without sending or receiving an ``abort`` message in exceptional
circumstances.

On ``epoch`` Arbor will commence the next epoch. Note that Arbor may expect the
last epoch to be shortened, ie when the total runtime is not a multiple of the
epoch length.

``Done`` signals the sending side is finished with the current simulation
period, i.e. the current call to ``simulation.run(T, dt)``. *May* cause the
receiving side to quit.

``Null`` does nothing, but reserved for future use, will currently not be sent
by Arbor.

We package these messsage as a C++ ``std::variant`` called ``ctrl_message`` in
``arbor/communication/remote.hpp`` alongside the ``exchange_ctrl`` method. This
will handle setting up the buffers, performing the actual transfer, and returns
the result as a ``ctrl_messge``. Handling the message is left to the
participating package.

**Important** This is a synchronous protocol which means an unannounced
termination of either side of the coupled simulators can lead to the other
getting stuck on a blocking call to MPI. This unlikely to cause issues in
scenarios where both sides are launched as a single job (eg via ``SLURM``), but
might do so where unrelated jobs are used.

Tying It All Together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While there is no requirement on doing, we strongly recommend to make use of the
facilities offered in ``arbor/communication/remote.hpp``, as does Arbor
internally. It should also be possible to interact with this protocol via ``C``
or other languages, if needed, as the infrastructure relies on byte-buffers and
numeric tags; the use of C++ types and variants on top is just an attempt to
make the interaction a bit safer and nicer. Refer to the ``remote.cpp`` example
on how they are used and the inline comments in ``remote.hpp``.

Terms and Definitions
---------------------

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
