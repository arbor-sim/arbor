#pragma once

#include <memory>
#include <string>

#include <arbor/spike.hpp>
#include <arbor/communication/gathered_vector.hpp>
#include <arbor/util/pp_util.hpp>

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

    std::vector<std::string> gather(std::string value, int root) const {
        return impl_->gather(value, root);
    }

private:
    struct interface {
        virtual gathered_vector<arb::spike>
            gather_spikes(const spike_vector& local_spikes) const = 0;
        virtual int id() const = 0;
        virtual int size() const = 0;
        virtual void barrier() const = 0;
        virtual std::string name() const = 0;

        ARB_PP_FOREACH(ARB_INTERFACE_COLLECTIVES_, ARB_COLLECTIVE_TYPES_)
        virtual std::vector<std::string> gather(std::string value, int root) const = 0;

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

        std::vector<std::string> gather(std::string value, int root) const override {
            return wrapped.gather(value, root);
        }

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

    int id() const { return 0; }

    int size() const { return 1; }

    template <typename T>
    T min(T value) const { return value; }

    template <typename T>
    T max(T value) const { return value; }

    template <typename T>
    T sum(T value) const { return value; }

    template <typename T>
    std::vector<T> gather(T value, int) const { return {std::move(value)}; }

    void barrier() const {}

    std::string name() const { return "local"; }
};

inline distributed_context::distributed_context():
    distributed_context(local_context())
{}

std::shared_ptr<distributed_context> dry_run_context(unsigned num_ranks, unsigned num_cells_per_rank);

// MPI context creation functions only provided if built with MPI support.

std::shared_ptr<distributed_context> mpi_context();

template <typename MPICommType>
std::shared_ptr<distributed_context> mpi_context(MPICommType);

} // namespace arb

