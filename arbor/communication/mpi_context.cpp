// Attempting to acquire an MPI context without MPI enabled will produce
// a link error.

#ifndef ARB_HAVE_MPI
#error "build only if MPI is enabled"
#endif

#include <string>
#include <vector>
#include <typeinfo>

#include <arbor/spike.hpp>
#include <arbor/util/unique_any.hpp>

#include "communication/mpi.hpp"
#include "distributed_context.hpp"
#include "label_resolution.hpp"

namespace arb {

// Throws arb::mpi::mpi_error if MPI calls fail.
struct mpi_context_impl {
    int size_ = -1;
    int rank_ = -1;
    MPI_Comm comm_ = MPI_COMM_NULL;

    explicit mpi_context_impl(MPI_Comm comm): comm_(comm) {
        size_ = mpi::size(comm_);
        rank_ = mpi::rank(comm_);
    }

    std::vector<arb::spike>
    remote_gather_spikes(const std::vector<arb::spike>& local_spikes) const {
        return {};
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
std::shared_ptr<distributed_context> make_mpi_context(MPI_Comm comm) {
    return std::make_shared<distributed_context>(mpi_context_impl(comm));
}

struct remote_context_impl {
    mpi_context_impl mpi_;
    MPI_Comm portal_ = MPI_COMM_NULL;

    explicit remote_context_impl(MPI_Comm comm, MPI_Comm portal):
        mpi_{comm}, portal_{portal} {}

    virtual std::vector<arb::spike>
    remote_gather_spikes(const std::vector<arb::spike>& local_spikes) const {
        return mpi::gather_all(local_spikes, portal_);
    }

    gathered_vector<arb::spike>
    gather_spikes(const std::vector<arb::spike>& local_spikes) const { return mpi_.gather_spikes(local_spikes); }

    gathered_vector<cell_gid_type>
    gather_gids(const std::vector<cell_gid_type>& local_gids) const { return mpi_.gather_gids(local_gids); }

    std::vector<std::vector<cell_gid_type>>
    gather_gj_connections(const std::vector<std::vector<cell_gid_type>>& local_connections) const {
        return mpi_.gather_gj_connections(local_connections);
    }

    cell_label_range gather_cell_label_range(const cell_label_range& local_ranges) const {
        return mpi_.gather_cell_label_range(local_ranges);
    }

    cell_labels_and_gids gather_cell_labels_and_gids(const cell_labels_and_gids& local_labels_and_gids) const {
        return mpi_.gather_cell_labels_and_gids(local_labels_and_gids);
    }

    template <typename T> std::vector<T> gather(T value, int root) const { return mpi_.gather(value, root); }
    std::string name() const { return "MPIRemote"; }
    int id() const { return mpi_.id(); }
    int size() const { return mpi_.size(); }
    template <typename T> T min(T value) const { return mpi_.min(value); }
    template <typename T> T max(T value) const { return mpi_.max(value); }
    template <typename T> T sum(T value) const { return mpi_.sum(value); }
    void barrier() const { mpi_.barrier(); }
};

template <>
std::shared_ptr<distributed_context> make_remote_context(MPI_Comm comm, MPI_Comm remote) {
    return std::make_shared<distributed_context>(remote_context_impl(comm, remote));
}

} // namespace arb
