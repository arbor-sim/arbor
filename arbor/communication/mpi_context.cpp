// Attempting to acquire an MPI context without MPI enabled will produce
// a link error.

#ifndef ARB_HAVE_MPI
#error "build only if MPI is enabled"
#endif

#include <string>
#include <vector>

#include <mpi.h>

#include <arbor/spike.hpp>
#include <arbor/util/scope_exit.hpp>

#include "communication/mpi.hpp"
#include "distributed_context.hpp"
#include "label_resolution.hpp"

#ifdef ARB_HAVE_HWLOC
#include <hwloc.h>
#endif

namespace arb {

// Throws arb::mpi::mpi_error if MPI calls fail.
struct mpi_context_impl {
    int size_;
    int rank_;
    MPI_Comm comm_;

    explicit mpi_context_impl(MPI_Comm comm, bool bind=false): comm_(comm) {
        size_ = mpi::size(comm_);
        rank_ = mpi::rank(comm_);

#define HWLOC(exp, msg) if (-1 == exp) throw arbor_internal_error(std::string{"HWLOC Process failed at: "} + msg);
#ifdef ARB_HAVE_HWLOC
        if (bind) {
            MPI_Comm local;
            MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &local);
            auto mpi_guard = util::on_scope_exit([&] { MPI_Comm_free(&local); });
            int local_rank = -1, local_size = -1;
            MPI_Comm_rank(local, &local_rank);
            MPI_Comm_size(local, &local_size);

            // Create the topology and ensure we don't leak it
            auto topology = hwloc_topology_t{};
            auto hlw_guard = util::on_scope_exit([&] { hwloc_topology_destroy(topology); });
            HWLOC(hwloc_topology_init(&topology), "Topo init");
            HWLOC(hwloc_topology_load(topology), "Topo load");
            // Fetch our current restrictions and apply them to our topology
            // NOTE: This is questionable, no?
            hwloc_cpuset_t proc_cpus{};
            HWLOC(hwloc_get_cpubind(topology, proc_cpus, HWLOC_CPUBIND_PROCESS), "Getting our cpuset.");
            HWLOC(hwloc_topology_restrict(topology, proc_cpus, 0), "Topo restriction.");
            // Extract the root object describing the full local node
            auto root = hwloc_get_root_obj(topology);
            // Allocate one set per rank on this node
            auto cpusets = std::vector<hwloc_cpuset_t>(local_size, {});
            // Distribute threads over topology, giving each of them as much private
            // cache as possible and keeping them locally in number order.
            HWLOC(hwloc_distrib(topology,
                                &root, 1,                        // single root for the full machine
                                cpusets.data(), cpusets.size(),  // one cpuset for each rank
                                INT_MAX,                         // maximum available level = Logical Cores
                                0),                              // No flags
                  "Distribute");
            // Now bind thread
            HWLOC(hwloc_set_cpubind(topology, cpusets[local_rank], HWLOC_CPUBIND_PROCESS),
                  "Binding");
        }
#endif
#undef HWLOC
    }

    gathered_vector<arb::spike>
    gather_spikes(const std::vector<arb::spike>& local_spikes) const {
        return mpi::gather_all_with_partition(local_spikes, comm_);
    }

    gathered_vector<cell_gid_type>
    gather_gids(const std::vector<cell_gid_type>& local_gids) const {
        return mpi::gather_all_with_partition(local_gids, comm_);
    }

    std::vector<std::vector<cell_gid_type>>
    gather_gj_connections(const std::vector<std::vector<cell_gid_type>>& local_connections) const {
        return mpi::gather_all(local_connections, comm_);
    }

    cell_label_range gather_cell_label_range(const cell_label_range& local_ranges) const {
        std::vector<cell_size_type> sizes;
        std::vector<cell_tag_type> labels;
        std::vector<lid_range> ranges;
        sizes  = mpi::gather_all(local_ranges.sizes(), comm_);
        labels = mpi::gather_all(local_ranges.labels(), comm_);
        ranges = mpi::gather_all(local_ranges.ranges(), comm_);
        return cell_label_range(sizes, labels, ranges);
    }

    cell_labels_and_gids gather_cell_labels_and_gids(const cell_labels_and_gids& local_labels_and_gids) const {
        auto global_ranges = gather_cell_label_range(local_labels_and_gids.label_range);
        auto global_gids = mpi::gather_all(local_labels_and_gids.gids, comm_);

        return cell_labels_and_gids(global_ranges, global_gids);
    }

    template <typename T>
    std::vector<T> gather(T value, int root) const {
        return mpi::gather(value, root, comm_);
    }

    std::string name() const { return "MPI"; }
    int id() const { return rank_; }
    int size() const { return size_; }

    template <typename T>
    T min(T value) const {
        return mpi::reduce(value, MPI_MIN, comm_);
    }

    template <typename T>
    T max(T value) const {
        return mpi::reduce(value, MPI_MAX, comm_);
    }

    template <typename T>
    T sum(T value) const {
        return mpi::reduce(value, MPI_SUM, comm_);
    }

    void barrier() const {
        mpi::barrier(comm_);
    }
};

template <>
std::shared_ptr<distributed_context> make_mpi_context(MPI_Comm comm, bool bind) {
    return std::make_shared<distributed_context>(mpi_context_impl(comm, bind));
}

} // namespace arb
