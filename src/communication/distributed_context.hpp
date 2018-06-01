#pragma once

#include <string>

#include <spike.hpp>
#include <communication/gathered_vector.hpp>
#include <util/pp_util.hpp>

#if defined(ARB_HAVE_MPI)
#   include "mpi_context.hpp"
#endif
#include "local_context.hpp"


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

#define ARB_COLLECTIVE_TYPES_ float, double, int, std::uint32_t, std::uint64_t

// distributed_context
//
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

    // default constructor uses a local context
    distributed_context(): distributed_context(local_context()) {}

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

        ARB_PP_FOREACH(ARB_INTERFACE_COLLECTIVES_, ARB_COLLECTIVE_TYPES_);
        virtual std::vector<std::string> gather(std::string value, int root) const = 0;

        virtual ~interface() {}
    };

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}

        gathered_vector<arb::spike>
        gather_spikes(const spike_vector& local_spikes) const {
            return wrapped.gather_spikes(local_spikes);
        }
        int id() const {
            return wrapped.id();
        }
        int size() const {
            return wrapped.size();
        }
        void barrier() const {
            wrapped.barrier();
        }
        std::string name() const {
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

} // namespace arb

