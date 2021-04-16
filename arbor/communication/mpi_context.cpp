// Attempting to acquire an MPI context without MPI enabled will produce
// a link error.

#ifndef ARB_HAVE_MPI
#error "build only if MPI is enabled"
#endif

#include <string>
#include <vector>

#include <mpi.h>

#include <arbor/label_resolver.hpp>
#include <arbor/spike.hpp>

#include "communication/mpi.hpp"
#include "distributed_context.hpp"
#include "util/partition.hpp"

namespace arb {

// Throws arb::mpi::mpi_error if MPI calls fail.
struct mpi_context_impl {
    int size_;
    int rank_;
    MPI_Comm comm_;

    explicit mpi_context_impl(MPI_Comm comm): comm_(comm) {
        size_ = mpi::size(comm_);
        rank_ = mpi::rank(comm_);
    }

    gathered_vector<arb::spike>
    gather_spikes(const std::vector<arb::spike>& local_spikes) const {
        return mpi::gather_all_with_partition(local_spikes, comm_);
    }

    gathered_vector<cell_gid_type>
    gather_gids(const std::vector<cell_gid_type>& local_gids) const {
        return mpi::gather_all_with_partition(local_gids, comm_);
    }

    cell_labeled_ranges gather_labeled_range(const cell_labeled_ranges& local_ranges) const {
        arb_assert(local_ranges.is_one_partition());
        std::vector<size_t> sizes, partitions;

        std::vector<cell_gid_type> gids   = mpi::gather_all(local_ranges.gids, comm_);
        std::vector<cell_tag_type> labels = mpi::gather_all(local_ranges.labels, comm_);
        std::vector<lid_range> ranges     = mpi::gather_all(local_ranges.ranges, comm_);

        sizes = mpi::gather_all(local_ranges.gids.size(), comm_);
        util::make_partition(partitions, sizes);

        return cell_labeled_ranges(gids, labels, ranges, partitions);
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

    template <typename T>
    std::vector<T> gather(T value, int root) const {
        return mpi::gather(value, root, comm_);
    }

    void barrier() const {
        mpi::barrier(comm_);
    }
};

template <>
std::shared_ptr<distributed_context> make_mpi_context(MPI_Comm comm) {
    return std::make_shared<distributed_context>(mpi_context_impl(comm));
}

} // namespace arb
