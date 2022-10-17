.. _external:

External Connectivity in Arbor
==============================

We describe how to connect an external simulator to Arbor using an MPI-based
protocol. The details are elaborated below, but the high-level flow is this

1. Handshake by passing an ``MPI_Intercommunicator``
2. Start a simulation

   1. Negotiate ``Dt`` and ``T``

       * Send suggestion and receive counter-offer using ``MPI_Send_recv``
       * Compute minimum of both, check validity
       * Send final value and receive counter-offer using ``MPI_Send_recv``
       * Proceed or abort

   2. Enter simulation loop

       * Advance ``t`` by ``Dt``
       * Send spike counts ``MPI_Allgather``
       * Send spikes ``MPI_Allgatherv``
       * Recv spike counts ``MPI_Allgather``
       * Recv spikes ``MPI_Allgatherv``
       * If ``t < T`` goto 2.2.

Initial Handshake
-----------------

The initial handshake with an external partner is commenced by adding an
``MPI_Intercommunicator`` to Arbor's context, e.g.

.. code:: cpp

   // Obtain an MPI_Intercommunicator, however you like (see below)
   auto inter = ...;
   // Create a context with the 'local' World
   auto ctx = arb::make_context({}, MPI_COMM_WORLD);
   // Now add the intercommunicator
   make_remote_connection(ctx, inter);

The construction of the ``MPI_Intercommunicator`` object is left to the consumer
of the Arbor library, common methods may include ``MPI_Comm_accept``,
``MPI_Comm_spawn``, and ``MPI_Comm_split``.

Arbor will inspect the ``MPI_Intercommunicator`` object during this phase, but
expects no further communication.

Definition: Root
''''''''''''''''

The *root task* on the intercommunicator is the task with rank ``0`` locally.
Thus, we have **two** roots, one on Arbor's and one on the remote side.

Starting a Simulation
---------------------

In Arbor a network simulation proceeds in discrete epochs of length ``Dt``
starting at the current time ``t`` until the final time ``T`` is reached. Each
epoch is integrated in time steps of width ``dt`` where ``dt ≤ Dt``. After each
epoch, spikes generated during said epoch ``[t, t + Dt)`` are sent over the
communicator.

Definition: Epoch
'''''''''''''''''

On Arbor's side an epoch is the global minimum of all synaptic delays divided by
two. This factor of two is due to the double buffering scheme used to hide
communication latencies. By definition we have ``Dt ≥ dt``.

Negotiating the Epoch Length
''''''''''''''''''''''''''''

When Arbor is integrated in a large context, the epoch length must be chosen
over the full context including Arbor. Thus, the global epoch  ``Dt = min(Dt_arb,
Dt_ext)``, while upholding the invariant ``Dt ≥ dt``.

Thus, a call to ``simulation::run(T, dt)`` will engage in the following protocol
before the first epoch

1. Send Arbor's current epoch using from its root task to the remote root ``MPI_Send_recv``
2. Consolidated local ``Dt_arb`` and remote ``Dt_ext`` into ``Dt = min(Dt_arb, Dt_ext)``.
   - If the resulting ``Dt`` does not suffice the invariants, we set ``Dt = -1``.
3. Send back ``Dt``, again using ``MPI_Send_recv``.
4. If we sent ``-1`` or received a value ``Dt_fin`` that is not the one we sent ``Dt``, we abort.
5. Enter the actual epoch loop.

This is done once per simulation since in Arbor the connectivity can change
between these calls.

Negotiate a Final Time
''''''''''''''''''''''

Using the same steps as above, the two partners agree on a final time ``T =
min(T_arb, T_ext)``, with the additional invariants of

- ``T`` is an integer multiple of ``Dt``.
- ``T ≥ Dt`` and by extension ``T ≥ dt``.
- ``T > t`` must be larger than the current time.

Again, the protocol may abort by sending ``T = -1`` at any step where a positive value is expected.
After ``t = T`` has been reached, the simulation stops and a new round of this negotiation starts.

Advancing the Time and Exchanging Spikes
----------------------------------------

During the call to ``simulation::run``, we execute multiple rounds of the following protocol.

1. Advance Arbor's internal state by ``Dt`` and collect all spikes generated during that time locally on each task ``i``. Call we the resulting buffer ``spikes_arb[i]`` and its length  ``n_spikes_arb[i]``.
   - Sort the local buffer according to the natural ordering on spikes (see below).
2. Announce the lengths to the external participants using ``MPI_Allgather`` over the intercommunicator.
   - Exactly one number is sent per local (Arbor) task
   - Call the resulting buffer on the remote side ``counts_arb_total``
3. Sent the local buffers across the intercommunicator using ``MPI_Allgatherv``
   - the remote side is expected to allocate enough space to receive as many spikes as announced in total during the last step
   - the displacements required by MPI are the left-scan-sum of the ``counts_arb_total`` array.
   - the concatenated global spike buffer is called ``spikes_arb_total`` for the following discussion.
4. Expect to receive the external spikes via the same protocol
   - Receive ``N`` counts via ``MPI_Allgather``.
   - Receive spikes from ``N`` domains according to the counts.

Thus in C++-ish pseudo-code with total disregard for the API definitions of
Arbor, MPI, and STL; describing the direction Arbor to the external partner.

.. code:: cpp

    // Run simulation, generate some spikes
    std::vector<spike_t> spikes = advance(Dt);
    spikes.sort();
    size_t count = spikes.size();

    // Collect all spike sizes; we have now one entry per source domain.
    std::vector<size_t> counts = MPI_Allgather(intercomm, count);
    // Scan-left sum
    size_t total = 0;
    for(auto& count: counts) {
       count += total;
       total = count;
    }
    // now, the total is the sum of all spike counts and counts[i] is the count
    // of all spikes on domains with id i and less.

    // Collect all spikes into one buffer on all processes simultaneously
    std::vector<spike> global_spikes = MPI_Allgatherv(intercomm, spikes, counts);
    // in particular, counts[i] gives the end of the spikes received from domain i

The MPI calls have been shortened to the sensible minimum, they do the expected
thing. After this exchange, the spikes are filtered and converted to events
concerning local cells and these are fed into the next call to ```advance``.
Then, the cycle starts anew. Note that in Arbor, the calls to MPI and
``advance`` execute concurrently.

Data Type Definition: Spike
'''''''''''''''''''''''''''

A spike is an ordered tuple ``(source:(gid:u32, lid:32), time:f64)``. Arbor's
interpretation of this tuple is this: A spike source is identified by its global
id ``gid``, which must be unique and its local id ``lid``, which is unique per
``gid``. Concretely, a ``gid`` might identify a cell in the network and the
``lid`` a particular spike detector assigned to that cell. As a cell can have
multiple detectors assigned, the ``lid`` is used in conjunction. Globally, the
pair ``(gid, lid)`` must be unique. In C++, we write

.. code:: cpp

   struct id_t {
     u32 gid, lid;
   };

   struct spike_t {
     id_t source;
     double time;
   };

Global Sorting
''''''''''''''
We sort spikes according to the natural lexicographic ordering defined by
``(gid, lid, time)`` before sending the locally generated spikes.

Arbor's domain partitioning algorithm ensures that domains are assigned to tasks
such that the domain ids are monotonically increasing with the tasks' ranks,
i.e. if rank ``n`` has domains with maximal ids ``[m, m+k)``, than rank ``n+1``
has ids greater than ``[m+k, m+k+l)`` and recursively. By definition domain
ids are contiguous.

Thus, to sort globally by source ``(domain, gid, lid)`` locally sorting followed
by ``MPI_Allgatherv`` is sufficient.
