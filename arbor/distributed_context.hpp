#pragma once

#include <memory>
#include <string>

#include <arbor/export.hpp>
#include <arbor/spike.hpp>
#include <arbor/util/pp_util.hpp>

#include "communication/gathered_vector.hpp"
#include "label_resolution.hpp"

namespace arb {

#define ARB_PUBLIC_COLLECTIVES_(T) \
    T min(T value) const { return impl_->min(value); }\
    T max(T value) const { return impl_->max(value); }\
    T sum(T value) const { return impl_->sum(value); }\
    std::vector<T> gather(T value, int root) const { return impl_->gather(value, root); }

#define ARB_INTERFACE_COLLECTIVES_(T) \
    virtual T min(T value) const = 0;\
    virtual T max(T value) const = 0;\
    virtual T sum(T value) const = 0;\
    virtual std::vector<T> gather(T value, int root) const = 0;

#define ARB_WRAP_COLLECTIVES_(T) \
    T min(T value) const override { return wrapped.min(value); }\
    T max(T value) const override { return wrapped.max(value); }\
    T sum(T value) const override { return wrapped.sum(value); }\
    std::vector<T> gather(T value, int root) const override { return wrapped.gather(value, root); }

#define ARB_COLLECTIVE_TYPES_ float, double, int, unsigned, long, unsigned long, long long, unsigned long long

// Defines the concept/interface for a distributed communication context.
//
// Uses value-semantic type erasure to define the interface, so that
// types that implement the interface can use duck-typing, without having
// to inherit from distributed_context.
//
// For the simplest example of a distributed_context implementation,
// see local_context, which is the default context.

class distributed_context {
public:
    using spike_vector = std::vector<arb::spike>;
    using gid_vector = std::vector<cell_gid_type>;
    using gj_connection_vector = std::vector<gid_vector>;

    // default constructor uses a local context: see below.
    distributed_context();

    template <typename Impl>
    distributed_context(Impl&& impl):
        impl_(new wrap<Impl>(std::forward<Impl>(impl)))
    {}

    distributed_context(distributed_context&& other) = default;
    distributed_context& operator=(distributed_context&& other) = default;

    gathered_vector<arb::spike> gather_spikes(const spike_vector& local_spikes) const {
        return impl_->gather_spikes(local_spikes);
    }

    gathered_vector<cell_gid_type> gather_gids(const gid_vector& local_gids) const {
        return impl_->gather_gids(local_gids);
    }

    gj_connection_vector gather_gj_connections(const gj_connection_vector& local_connections) const {
        return impl_->gather_gj_connections(local_connections);
    }

    cell_label_range gather_cell_label_range(const cell_label_range& local_ranges) const {
        return impl_->gather_cell_label_range(local_ranges);
    }

    cell_labels_and_gids gather_cell_labels_and_gids(const cell_labels_and_gids& local_labels_and_gids) const {
        return impl_->gather_cell_labels_and_gids(local_labels_and_gids);
    }

    std::vector<std::string> gather(std::string value, int root) const {
        return impl_->gather(value, root);
    }

    int id() const {
        return impl_->id();
    }

    int size() const {
        return impl_->size();
    }

    void barrier() const {
        impl_->barrier();
    }

    std::string name() const {
        return impl_->name();
    }

    ARB_PP_FOREACH(ARB_PUBLIC_COLLECTIVES_, ARB_COLLECTIVE_TYPES_);

private:
    struct interface {
        virtual gathered_vector<arb::spike>
            gather_spikes(const spike_vector& local_spikes) const = 0;
        virtual gathered_vector<cell_gid_type>
            gather_gids(const gid_vector& local_gids) const = 0;
        virtual gj_connection_vector
            gather_gj_connections(const gj_connection_vector& local_connections) const = 0;
        virtual cell_label_range
            gather_cell_label_range(const cell_label_range& local_ranges) const = 0;
        virtual cell_labels_and_gids
            gather_cell_labels_and_gids(const cell_labels_and_gids& local_labels_and_gids) const = 0;
        virtual std::vector<std::string>
            gather(std::string value, int root) const = 0;
        virtual int id() const = 0;
        virtual int size() const = 0;
        virtual void barrier() const = 0;
        virtual std::string name() const = 0;

        ARB_PP_FOREACH(ARB_INTERFACE_COLLECTIVES_, ARB_COLLECTIVE_TYPES_)

        virtual ~interface() {}
    };

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}

        gathered_vector<arb::spike>
        gather_spikes(const spike_vector& local_spikes) const override {
            return wrapped.gather_spikes(local_spikes);
        }
        gathered_vector<cell_gid_type>
        gather_gids(const gid_vector& local_gids) const override {
            return wrapped.gather_gids(local_gids);
        }
        std::vector<std::vector<cell_gid_type>>
        gather_gj_connections(const gj_connection_vector& local_connections) const override {
            return wrapped.gather_gj_connections(local_connections);
        }
        cell_label_range
        gather_cell_label_range(const cell_label_range& local_ranges) const override {
            return wrapped.gather_cell_label_range(local_ranges);
        }
        cell_labels_and_gids
        gather_cell_labels_and_gids(const cell_labels_and_gids& local_labels_and_gids) const override {
            return wrapped.gather_cell_labels_and_gids(local_labels_and_gids);
        }
        std::vector<std::string>
        gather(std::string value, int root) const override {
            return wrapped.gather(value, root);
        }
        int id() const override {
            return wrapped.id();
        }
        int size() const override {
            return wrapped.size();
        }
        void barrier() const override {
            wrapped.barrier();
        }
        std::string name() const override {
            return wrapped.name();
        }

        ARB_PP_FOREACH(ARB_WRAP_COLLECTIVES_, ARB_COLLECTIVE_TYPES_)

        Impl wrapped;
    };

    std::unique_ptr<interface> impl_;
};

struct local_context {
    gathered_vector<arb::spike>
    gather_spikes(const std::vector<arb::spike>& local_spikes) const {
        using count_type = typename gathered_vector<arb::spike>::count_type;
        return gathered_vector<arb::spike>(
            std::vector<arb::spike>(local_spikes),
            {0u, static_cast<count_type>(local_spikes.size())}
        );
    }
    gathered_vector<cell_gid_type>
    gather_gids(const std::vector<cell_gid_type>& local_gids) const {
        using count_type = typename gathered_vector<cell_gid_type>::count_type;
        return gathered_vector<cell_gid_type>(
                std::vector<cell_gid_type>(local_gids),
                {0u, static_cast<count_type>(local_gids.size())}
        );
    }
    std::vector<std::vector<cell_gid_type>>
    gather_gj_connections(const std::vector<std::vector<cell_gid_type>>& local_connections) const {
        return local_connections;
    }
    cell_label_range
    gather_cell_label_range(const cell_label_range& local_ranges) const {
        return local_ranges;
    }
    cell_labels_and_gids
    gather_cell_labels_and_gids(const cell_labels_and_gids& local_labels_and_gids) const {
        return local_labels_and_gids;
    }
    template <typename T>
    std::vector<T> gather(T value, int) const {
        return {std::move(value)};
    }

    int id() const { return 0; }

    int size() const { return 1; }

    template <typename T>
    T min(T value) const { return value; }

    template <typename T>
    T max(T value) const { return value; }

    template <typename T>
    T sum(T value) const { return value; }

    void barrier() const {}

    std::string name() const { return "local"; }
};

inline distributed_context::distributed_context():
    distributed_context(local_context())
{}

using distributed_context_handle = std::shared_ptr<distributed_context>;

inline
distributed_context_handle make_local_context() {
    return std::make_shared<distributed_context>();
}

ARB_ARBOR_API distributed_context_handle make_dry_run_context(unsigned num_ranks, unsigned num_cells_per_rank);

// MPI context creation functions only provided if built with MPI support.
template <typename MPICommType>
distributed_context_handle make_mpi_context(MPICommType);

} // namespace arb

