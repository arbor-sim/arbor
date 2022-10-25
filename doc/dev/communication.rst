.. _communication:

Communication in Arbor
======================

Communication between cells (and thus cell groups) is facilitated using discrete
events which we call `spikes` in analogy to the underlying neurobiological
process. Spikes are the only way to communicate between different cell kinds, eg
cable cells and integrate-and-fire. In accordance to current theory, spikes are
reduced to a simple time value. The connection is abstracted into a weight and a
delay; modelling all axonal processes. While this may seem crude, it is a well
supported model and commonly used in neuronal simulations.

Connections are formed between sources (cable cell: threshold detectors) and targets
(cable cell: synapses). During runtime, events from all sources are concatenated
on all MPI ranks using ``Allgatherv`` and targets are responsible for selecting
events they have subscribed to. This is optimised for by sorting events locally
(by source) and relying on the process layout to convert this into a *globally*
sorted array.

Exchange of Spikes
==================

We will start by discussing the exchange of spikes, an extended summary of the
full communicator class can be found in the next section. Communication of
spikes is facilitated through the ``communicator::exchange`` method. which is

.. code-block:: c++

   // aka cell_member_type, lexicographic order provided
   struct id_type {
       gid_type gid;
       lid_type index;
   };

   struct spike {
       id_type source;
       double time;
   };

   using spike_vec = std::vector<spike>;

   // From gathered_vector
   template <typename T>
   struct gathered_vector {
       std::vector<T> values_;
       std::vector<int> partition_;
   };

   using g_spike_vec = gathered_vector<spike>;

   g_spike_vec communicator::exchange(spike_vec local_spikes) { // Take by value, since we modify (sort) anyhow.
       // sort the spikes in ascending order of source gid
       util::sort_by(local_spikes, [](spike s){ return s.source; });
       // global all-to-all to gather a local copy of the global spike list on each node.
       auto global_spikes = distributed_->gather_spikes(local_spikes);
       num_spikes_ += global_spikes.size();
       return global_spikes;
   }

   // From mpi_context
   g_spike_vec gather_spikes(const spike_ve& local_spikes) const {
       return mpi::gather_all_with_partition(local_spikes, comm_);
   }

   // From mpi
   g_spike_vec gather_all_with_partition(const spike_vec& values, MPI_Comm comm) {
       using traits = mpi_traits<spike>;
       // Collect all sender's counts, scale to bytes
       std::vector<int> counts = gather_all(int(values.size()), comm);
       for (auto& c: counts) c *= traits::count();
       // Scan-left to get offsets
       std::vector<int> displs;
       util::make_partition(displs, counts);
       // Allocate recv buffer and exchange
       spike_vec buffer(displs.back()/traits::count());
       MPI_Allgatherv(// send buffer  count to send       MPI datatype of spike
                      values.data(),  counts[rank(comm)], traits::mpi_type(),
                      // recv buffer  count of each sender offset of senders  MPI datatype of spike
                      buffer.data(),  counts.data(),       displs.data(),     traits::mpi_type(),
                      comm);
       // Scale back to sizeof(spike)
       for (auto& d: displs) d /= traits::count();
       return {std::move(buffer), std::move(displs)};
   }

Note that these snippets have been simplified and shortened from the
actual code, they are intended for education only.

After ``exchange`` is done, each process has received an object like
this

.. code-block:: c++

   {
     spikes:  { {s0, t0}, ...},
     offsets: { 0, ... }
   }

Now ``spikes`` is the array of all spikes during the last *epoch* where
each sub-array of spikes is sorted, ie between ``offsets[ix]`` and
``offest[ix+1]``. The ``offsets`` array has a length of MPI task count
and its ``i``'th element gives the position of the first spike sent by
task ``i``.

.. _event_distribution:

Distribution of Events to Targets
=================================

Having received the generated spikes, the concatenated data is converted
into events on each local cell group. This is done asynchronously with
computation of the next cell state. In ``simulation.cpp`` we find

.. code-block:: c++

   auto exchange = [this](epoch prev) {
       // Collate locally generated spikes.
       auto all_local_spikes = local_spikes(prev.id).gather();
       // Gather generated spikes across all ranks.
       auto global_spikes = communicator_.exchange(all_local_spikes);
       // Append events formed from global spikes to per-cell pending event queues.
       communicator_.make_event_queues(global_spikes, pending_events_);
   };

which uses this

.. code-block:: c++

   // Check each global spike in turn to see it generates local events.
   // If so, make the events and insert them into the appropriate event list.
   //
   // Takes reference to a vector of event lists as an argument, with one list
   // for each local cell group. On completion, the events in each list are
   // all events that must be delivered to targets in that cell group as a
   // result of the global spike exchange, plus any events that were already
   // in the list.
   void make_event_queues(const gathered_vector<spike>& global_spikes,
                          std::vector<pse_vector>& queues) {
       // Predicate for partitioning
       struct spike_pred {
           bool operator()(const spike& spk, const cell_member_type& src) { return spk.source < src; }
           bool operator()(const cell_member_type& src, const spike& spk) { return src < spk.source; }
       };

       const auto& sp = global_spikes.partition();
       for (auto dom: util::make_span(num_domains_)) {
           // Fetch connections and spikes per integration domain
           auto cons = util::subrange_view(connections_,           connection_part_[dom], connection_part_[dom + 1]);
           auto spks = util::subrange_view(global_spikes.values(), sp[dom],               sp[dom + 1]);
           auto sp = spks.begin(), se = spks.end();
           auto cn = cons.begin(), ce = cons.end();
           // We have a choice of whether to walk spikes or connections:
           // i.e., we can iterate over the spikes, and for each spike search
           // the for connections that have the same source; or alternatively
           // for each connection, we can search the list of spikes for spikes
           // with the same source.
           //
           // We iterate over whichever set is the smallest, which has
           // complexity of order max(S log(C), C log(S)), where S is the
           // number of spikes, and C is the number of connections.
           if (cons.size() < spks.size()) {
               while (cn != ce && sp != ce) {
                   auto src = cn->source();           // Source for connection
                   auto cix = cn->index_on_domain();  // Queue index for connection
                   // Given a source src split the range [sp, spks.end) into a pair sources=[l, h]
                   // st  *l is the last element not smaller than src
                   // and *h is the first element greater than src.
                   // 'Greater' and 'smaller' are defined via the predicate above.
                   // The range [sp, spks.end) must be (partially) ordered wrt the predicate.
                   auto sources = std::equal_range(sp, se, src, spike_pred());
                   // Consequently, the range returned is the range of equal spike sources,
                   // we pick out ours and add all of them to the appropriate queue.
                   for (auto s: util::make_range(sources)) queues[cix].push_back(cn->make_event(s));
                   // now, move to next
                   sp = sources.first;
                   ++cn;
               }
           }
           else {
               while (cn != ce && sp != se) {
                   auto targets = std::equal_range(cn, ce, sp->source);
                   for (auto c: util::make_range(targets)) queues[c.index_on_domain()].push_back(c.make_event(*sp));
                   cn = targets.first;
                   ++sp;
               }
           }
       }
   }

After ``make_event_queues`` there is one queue per cell and each queue
is filled with a time ordered list of events for that cell. We now need
to understand the actual connection table stored in

.. code-block:: c++

   struct connection {
       spike_event make_event(const spike& s) {
           return { destination_, s.time + delay_, weight_};
       }

       cell_member_type source;
       cell_lid_type destination;
       float weight;
       float delay;
       cell_size_type index_on_domain;
   };

   struct communicator {
       // [...]
       cell_size_type num_domains_;
       std::vector<connection> connections_;
       std::vector<cell_size_type> connection_part_;
       // [...]
   };

The ``connections`` vector is a list of connections partitioned by the
domain (as in domain decomposition) of their source's ``gid``, while
``connection_part`` stores the partioning indices.

Building the Connection Table
=============================

The table of connections on the local rank is built during the construction of
the ``communicator`` object

.. code-block:: c++

   communicator::communicator(const recipe& rec,
                              const domain_decomposition& dom_dec,
                              const label_resolution_map& source_resolution_map,
                              const label_resolution_map& target_resolution_map,
                              execution_context& ctx);

After that process,

.. code-block:: c++

   struct communicator {
       // ...
       std::vector<connection> connections_;
       std::vector<cell_size_type> connection_part_;
   };

will contain all connections in ``connections_`` partitioned by the
domain of the source's ``gid`` in ``dom_dec``. Beginnings of the
respective partitions are pointed to by the indices in
``connection_part_``.

The algorithm for building is slightly obscured by caching and the use
of labels and resolving them via ``target_/source_resolution_map`` to
local ids on the respective source and target cells.

.. note::

   The ``label_resolution_map`` class is used to translate from labels at the
   user facing API layers to Arbor's internal mappings in the vein of
   ``(cell_gid, item_offset)``, where ``item_offset`` is an automatically
   assigned integer ID. Textual labels are created by calls to ``place``
   as in this example

   .. code-block:: c++

      auto d = arb::decor{};
      d.place("..."_ls, arb::synapse{"..."}, "synapse-label");

The construction is performed in-place

.. code-block:: c++

    // Allocate space for our connections
    connections_.resize(n_cons);
    // We have pre-computed `src_counts`, connection_part_ will now hold the starting indices
    // of each `domain`.
    util::make_partition(connection_part_, src_counts);
    // Copy, as we use this as the list of the currently available next free target slots in
    // `connections_`
    auto offsets = connection_part_;
    auto target_resolver = resolver(&target_resolution_map);
    auto src_domain = src_domains.begin();
    for (const auto& cell: gid_infos) {
        auto index = cell.index_on_domain;
        auto source_resolver = resolver(&source_resolution_map);
        for (const auto& c: cell.conns) {
            // Compute index representation of labels
            auto src_lid = source_resolver.resolve(c.source);
            auto tgt_lid = target_resolver.resolve({cell.gid, c.dest});
            // Get offset of current source and bump to next free slot
            auto offset  = offsets[*src_domain]++;
            // Write connection info into slot
            connections_[offset] = {{c.source.gid, src_lid}, tgt_lid, c.weight, c.delay, index};
            // Next source domain
            ++src_domain;
        }
    }
    // Now
    // * all slots in `connections_` are filled.
    // * `offsets` points at the ends of each partition.


Next, each *partition* is sorted independently according to their
source's ``gid``.
