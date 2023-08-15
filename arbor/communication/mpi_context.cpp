// Attempting to acquire an MPI context without MPI enabled will produce
// a link error.

#ifndef ARB_HAVE_MPI
#error "build only if MPI is enabled"
#endif

#include <memory>
#include <string>
#include <vector>

#include <arbor/spike.hpp>
#include <arbor/communication/remote.hpp>
#include <arbor/util/scope_exit.hpp>

#include "communication/mpi.hpp"
#include "distributed_context.hpp"
#include "label_resolution.hpp"
#include "affinity.hpp"

namespace arb {

// Throws arb::mpi::mpi_error if MPI calls fail.
struct mpi_context_impl {
    int size_ = -1;
    int rank_ = -1;
    MPI_Comm comm_ = MPI_COMM_NULL;

    explicit mpi_context_impl(MPI_Comm comm, bool bind=false): comm_(comm) {
        size_ = mpi::size(comm_);
        rank_ = mpi::rank(comm_);
        if (bind) {
            MPI_Comm local;
            MPI_OR_THROW(MPI_Comm_split_type, comm, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &local);
            auto mpi_guard = util::on_scope_exit([&] { MPI_Comm_free(&local); });
            set_affinity(mpi::rank(local),
                         mpi::size(local),
                         affinity_kind::process);
        }
    }

    std::vector<spike>
    remote_gather_spikes(const std::vector<spike>& local_spikes) const {
        return {};
    }

    gathered_vector<spike>
    gather_spikes(const std::vector<spike>& local_spikes) const {
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

    std::vector<std::size_t> gather_all(std::size_t value) const {
        return mpi::gather_all(value, comm_);
    }

    distributed_request send_recv_nonblocking(std::size_t recv_count,
        void* recv_data,
        int source_id,
        std::size_t send_count,
        const void* send_data,
        int dest_id,
        int tag) const {

        // Return dummy request of nothing to do
        if (!recv_count && !send_count)
            return distributed_request{
                std::make_unique<distributed_request::distributed_request_interface>()};
        if(recv_count && !recv_data)
            throw arbor_internal_error(
                "send_recv_nonblocking: recv_data is null.");

        if(send_count && !send_data)
            throw arbor_internal_error(
                "send_recv_nonblocking: send_data is null.");

        if (recv_data == send_data)
            throw arbor_internal_error(
                "send_recv_nonblocking: recv_data and send_data must not be the same.");

        auto recv_requests = mpi::irecv(recv_count, recv_data, source_id, tag, comm_);
        auto send_requests = mpi::isend(send_count, send_data, dest_id, tag, comm_);

        struct mpi_send_recv_request : public distributed_request::distributed_request_interface {
            std::vector<MPI_Request> recv_requests, send_requests;

            mpi_send_recv_request(std::vector<MPI_Request> recv_requests,
                std::vector<MPI_Request> send_requests):
                recv_requests(std::move(recv_requests)),
                send_requests(std::move(send_requests)) {}

            void finalize() override {
                if (!recv_requests.empty()) {
                    mpi::wait_all(std::move(recv_requests));
                }

                if (!send_requests.empty()) {
                    mpi::wait_all(std::move(send_requests));
                }
            };

            ~mpi_send_recv_request() override { this->finalize(); }
        };

        return distributed_request{
            std::unique_ptr<distributed_request::distributed_request_interface>(
                new mpi_send_recv_request{std::move(recv_requests), std::move(send_requests)})};
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
    void remote_ctrl_send_continue(const epoch&) const {}
    void remote_ctrl_send_done() const {}
};

template <>
std::shared_ptr<distributed_context> make_mpi_context(MPI_Comm comm, bool bind) {
    return std::make_shared<distributed_context>(mpi_context_impl(comm, bind));
}

struct remote_context_impl {
    mpi_context_impl mpi_;
    MPI_Comm portal_ = MPI_COMM_NULL;

    explicit remote_context_impl(MPI_Comm comm, MPI_Comm portal):
        mpi_{comm}, portal_{portal} {}

    std::vector<spike>
    remote_gather_spikes(const std::vector<spike>& local_spikes) const {
        // Static sanity checks to ensure we can bit-cast spike <> remote::spike
        static_assert((sizeof(cell_member_type) == sizeof(remote::arb_cell_id))
                   && (offsetof(cell_member_type, gid) == offsetof(remote::arb_cell_id, gid))
                   && (offsetof(cell_member_type, index) == offsetof(remote::arb_cell_id, lid))
                   && std::is_same_v<cell_gid_type, remote::arb_gid_type>
                   && std::is_same_v<cell_lid_type, remote::arb_lid_type>,
                      "Remote cell identifier has diverged from Arbor's internal type.");
        static_assert((sizeof(spike) == sizeof(remote::arb_spike))
                    && (offsetof(spike, source) == offsetof(remote::arb_spike, source))
                    && (offsetof(spike, time) == offsetof(remote::arb_spike, time))
                    && std::is_same_v<time_type, remote::arb_time_type>,
                      "Remote spike type is diverged from Arbor's internal type.");
        return mpi::gather_all(local_spikes, portal_);
    }

    gathered_vector<spike>
    gather_spikes(const std::vector<spike>& local_spikes) const { return mpi_.gather_spikes(local_spikes); }

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
    void remote_ctrl_send_continue(const epoch& e) const { remote::exchange_ctrl(remote::msg_epoch{e.t0, e.t1}, portal_); }
    void remote_ctrl_send_done() const { remote::exchange_ctrl(remote::msg_done{}, portal_); }
};

template <>
std::shared_ptr<distributed_context> make_remote_context(MPI_Comm comm, MPI_Comm remote) {
    int is_inter = 0;
    MPI_Comm_test_inter(remote, &is_inter);
    if (!is_inter) throw mpi_inter_comm_required{};
    return std::make_shared<distributed_context>(remote_context_impl(comm, remote));
}

} // namespace arb
