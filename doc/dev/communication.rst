.. _communication:

Communication in Arbor
======================

Communication between cells (and thus cell groups) is facilitated using discrete
events which we call `spikes` in analogy to the underlying neurobiological
process. Spikes are the only way to communicate between different cell kinds, e.g.,
cable cells and integrate-and-fire. In accordance with current theory, spikes are
reduced to a simple time value. The connection is abstracted into a weight and a
delay; modelling all axonal processes. While this may seem crude, it is a well-supported model and commonly used in neuronal simulations.

Connections are formed between sources (cable cell: threshold detectors) and
targets (cable cell: synapses). During runtime, spikes generated on each MPI
rank are sorted locally by source and grouped according to their destination
ranks. Using MPI_Alltoallv, spike data is sent only to ranks that have
targets connected to their sources, avoiding unnecessary data transfers.
The received spikes are partitioned by sending rank and remain sorted by
source within each partition, allowing them to be efficiently matched with
local connections and converted into events.

Exchange of Spikes
==================

The exchange of spikes is handled by ``communicator::exchange``.
This method sorts locally generated spikes, distributes them to
the MPI ranks that require them, and collects spikes received
from external simulators.

The following simplified definitions and implementation illustrate
this process:

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

    template <typename T>
    struct gathered_vector {
        using count_type = unsigned;

        std::vector<T> values_;
        std::vector<count_type> partition_;
    };

    struct spikes {
        gathered_vector<spike> from_local;
        std::vector<spike> from_remote;
    };

   communicator::spikes communicator::exchange(spike_vec& local_spikes) {
        // sort the spikes in ascending order of source gid
        util::sort_by(local_spikes, [](spike s){ return s.source; });

        // Count all spikes generated across MPI ranks.
        num_local_spikes_ = ctx_->distributed->sum(local_spikes.size());
        num_spikes_ += num_local_spikes_;

        // Group spikes according to their destination ranks.
        auto spikes_per_rank = generate_all_to_all_vector(
           local_spikes, src_ranks_, num_domains_);

        // Exchange spikes only with the ranks that need them.
        auto received_spikes =
           ctx_->distributed->all_to_all_spikes(spikes_per_rank);

        // Exchange spikes with external simulators.
        // Optional filtering of outgoing spikes is omitted here.
        auto remote_spikes =
            ctx_->distributed->remote_gather_spikes(local_spikes);

        // Mark external sources and sort the received spikes.
        std::for_each(remote_spikes.begin(), remote_spikes.end(),
                      [](spike& s) { s.source = global_cell_of(s.source); });
        std::sort(remote_spikes.begin(), remote_spikes.end());

        return {std::move(received_spikes), std::move(remote_spikes)};
   }

Before exchanging spikes, ``generate_all_to_all_vector`` organizes
the locally generated spikes into groups according to their destination
ranks.

During connection setup, each rank identifies the spike sources required
by its local connections. This information is exchanged using
``all_to_all_gids_domains`` to construct ``src_ranks_``, which maps each
local spike source to the ranks that require its spikes.

Duplicate source identifiers are removed before this exchange.
Consequently, each generated spike is sent only once to each
destination rank, even if that rank has multiple connections
originating from the same source.

The function first counts the spikes destined for each rank and computes
the corresponding offsets. It then copies the spikes into a contiguous
buffer partitioned by destination rank.

.. code-block:: c++

    // Simplified version of generate_all_to_all_vector
    gathered_vector<spike> generate_all_to_all_vector(
            const std::vector<spike>& spikes,
            const std::unordered_map<cell_member_type, std::vector<cell_size_type>>& src_ranks,
            std::size_t num_domains) {

        // Count outgoing spikes per destination rank.
        using count_type = gathered_vector<spike>::count_type;
        std::vector<count_type> offsets(num_domains + 1, 0);
        for (const auto& spk: spikes) {
            auto ranks = src_ranks.find(spk.source);
            if (ranks != src_ranks.end()) {
                for (auto rank: ranks->second) {
                    ++offsets[rank + 1];
                }
            }
        }

        // Compute the offsets of each rank's partition.
        std::partial_sum(offsets.begin(), offsets.end(),
                        offsets.begin());

        // Allocate and fill the partitioned send buffer.
        std::vector<spike> spikes_per_rank(offsets.back());
        auto positions = offsets;
        for (const auto& spk: spikes) {
            auto ranks = src_ranks.find(spk.source);
            if (ranks != src_ranks.end()) {
                for (auto rank: ranks->second) {
                    spikes_per_rank[positions[rank]++] = spk;
                }
            }
        }

        return {std::move(spikes_per_rank),
                std::move(offsets)};
    }

The resulting ``gathered_vector`` contains a contiguous array of
spikes and a partition describing the range assigned to each
destination rank. Because local spikes were previously sorted by
source, their order is preserved within each partition.

The actual exchange is performed through ``all_to_all_spikes``,
which calls ``mpi::all_to_all_with_partition``.

First, ``MPI_Alltoall`` exchanges the send counts between ranks,
allowing each rank to determine the amount of data it will receive.
These counts are used to compute the receive displacements and
allocate the receive buffer.

Then, ``MPI_Alltoallv`` exchanges the spike data using the calculated
counts and displacements.

.. code-block:: c++

    // Exchange the number of elements sent to each rank.
    MPI_Alltoall(
        send_counts.data(), 1, MPI_INT,
        recv_counts.data(), 1, MPI_INT,
        comm);

    // Compute receive displacements and allocate the receive buffer.
    // ...

    // Exchange the spike data.
    MPI_Alltoallv(
        send_buffer.data(),
        send_counts.data(),
        send_displs.data(),
        traits::mpi_type(),
        recv_buffer.data(),
        recv_counts.data(),
        recv_displs.data(),
        traits::mpi_type(),
        comm);

Note that these snippets have been simplified and shortened from the
actual code. They are intended for educational purposes only.

After the exchange is complete, each rank receives a
``communicator::spikes`` object containing two components:
``from_local`` and ``from_remote``.

``from_local`` contains the spikes received from other Arbor MPI ranks,
including the local rank when applicable. It is represented by a
gathered_vector containing the received spikes and their partition
offsets.

For example, with three MPI ranks, the result might look like:

.. code-block:: text

    from_local.values():
        [spike_from_rank_0,
        spike_from_rank_0,
        spike_from_rank_2]

    from_local.partition():
        [0, 2, 2, 3]

In this example, two spikes were received from rank 0, none from
rank 1, and one from rank 2.

The partition array contains num_domains + 1 elements.
For each rank i, the range between partition[i] and
partition[i+1] identifies the spikes received from that rank.

Each partition is sorted by source, allowing efficient matching
between received spikes and local connections during event generation.

The ``from_remote`` vector contains spikes received from external
simulators when co-simulation is enabled. These spikes are handled
separately from the internal MPI spike exchange.

.. _event_distribution:

Distribution of Events to Targets
=================================

Having received the generated spikes, each MPI rank converts them into
events for its local target cells. This is done asynchronously with
the computation of the next cell state. In ``simulation.cpp`` we find:

.. code-block:: c++

    auto exchange = [this](epoch prev) {
        // Collate locally generated spikes.
        auto all_local_spikes = local_spikes(prev.id).gather();

        // Exchange spikes with ranks that have subscribed targets.
        auto spikes = communicator_.exchange(all_local_spikes);

        // Append events formed from received spikes to pending event queues.
        communicator_.make_event_queues(spikes, pending_events_);
    };

which uses this

.. code-block:: c++

    void communicator::make_event_queues(communicator::spikes& spikes,
                                     std::vector<pse_vector>& queues) {
        arb_assert(queues.size()==num_local_cells_);
        const auto& sp = spikes.from_local.partition();
        const auto& cp = connection_part_;
        for (auto dom: util::make_span(num_domains_)) {
            append_events_from_domain(connections_, cp[dom], cp[dom+1],
                                      util::subrange_view(spikes.from_local.values(), sp[dom], sp[dom+1]),
                                      queues);
        }
        num_local_events_ = util::sum_by(queues, [](const auto& q) {return q.size();}, num_local_events_);
        // Now that all local spikes have been processed; consume the remote events coming in.
        // - turn all gids into externals
        std::for_each(spikes.from_remote.begin(), spikes.from_remote.end(),
                      [](auto& s) { s.source = global_cell_of(s.source); });
        append_events_from_domain(ext_connections_, 0, ext_connections_.size(), spikes.from_remote, queues);
    }

For each MPI rank, ``make_event_queues`` selects the corresponding
partitions of received spikes and local connections. Since both are
sorted by source, ``append_events_from_domain`` can efficiently
match them by traversing the sorted sequences.

For each matching spike and connection, an event is created using
the connection's destination, delay and weight. The relevant operation
inside append_events_from_domain can be summarized as follows:

.. code-block:: c++

    // For each matching spike and connection.
    auto cell = cons.idx_on_domain[cn];
    auto dst = cons.dests[cn];
    auto del = cons.delays[cn];
    auto wgt = cons.weights[cn];

    queues[cell].emplace_back(dst, sp->time + del, wgt);

The connection information is stored in ``connection_list``, with
connections partitioned by their source rank. The construction of
this table is described in the next section.

After ``make_event_queues``, there is one pending event queue per
local cell. These queues are subsequently sorted by time, target
and weight in ``simulation.cpp`` before being merged with other events.

Building the Connection Table
=============================

The table of connections on the local rank is built by
the ``communicator::update_connections`` method:

.. code-block:: c++

   void communicator::update_connections(const recipe& rec,
                                         const domain_decomposition_ptr dom_dec,
                                         const label_resolution_map& source_resolution_map,
                                         const label_resolution_map& target_resolution_map);

The communicator stores the connections using:

.. code-block:: c++

    class communicator {
        //...
        connection_list connections_;
        std::vector<cell_size_type> connection_part_;
    };

``connection_list`` stores connection attributes in separate arrays:

.. code-block:: c++

    struct connection_list {
        std::vector<cell_size_type> idx_on_domain;
        std::vector<cell_member_type> srcs;
        std::vector<cell_lid_type> dests;
        std::vector<float> weights;
        std::vector<float> delays;
    };

The ``connection_part_`` vector identifies the boundaries of each
source rank's connections in ``connections_``.

Connections are collected from the recipe and generated network
descriptions. Source and target labels are resolved to their internal
identifiers using the corresponding resolution maps.

Each connection is assigned to the MPI rank that owns its source
cell and grouped into ``connections_by_src_domain`` before sorting
and storing in ``connection_list``.

.. note::

   The ``label_resolution_map`` class is used to translate from labels at the
   user-facing API layers to Arbor's internal mappings in the vein of
   ``(cell_gid, item_offset)``, where ``item_offset`` is an automatically
   assigned integer ID. Textual labels are created by calls to ``place``
   as in this example

   .. code-block:: c++

      auto d = arb::decor{};
      d.place("..."_ls, arb::synapse{"..."}, "synapse-label");

Connections are grouped by the MPI rank that owns their source cell.
Within each group, connections are sorted by their source identifier.

.. code-block:: c++

    // Sort connections independently for each source rank.
    for (auto& cons: connections_by_src_domain) {
        util::sort(cons);
    }

    // Compute partition offsets.
    reset_partition(connections_by_src_domain, connection_part_);

    // Store connections in separate arrays.
    connections_.clear();
    connections_.reserve(n_con);
    connections_.make(connections_by_src_domain);

The ``connection_part_`` vector identifies the boundaries of each
source rank's connections in ``connections_``. This organization allows
the received spikes to be efficiently matched with their corresponding
connections during event generation.
