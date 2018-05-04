#pragma once

#include <string>

#include <spike.hpp>
#include <communication/gathered_vector.hpp>

#if defined(ARB_HAVE_MPI)
#   include "mpi_context.hpp"
#endif
#include "serial_context.hpp"


namespace arb {

enum class global_policy_kind {serial, mpi, dryrun};

inline std::string to_string(global_policy_kind k) {
    if (k == global_policy_kind::mpi) {
        return "MPI";
    }
    if (k == global_policy_kind::dryrun) {
        return "dryrun";
    }
    return "serial";
}

#define PUBLIC_REDUCE(T) \
    T min(T value) const { return impl_->min(value); }\
    T max(T value) const { return impl_->max(value); }\
    T sum(T value) const { return impl_->sum(value); }
#define PUBLIC_GATHER(T) std::vector<T> gather(T value, int root) const { return impl_->gather(value, root); }

#define INTERFACE_REDUCE(T) \
    virtual T min(T value) const = 0;\
    virtual T max(T value) const = 0;\
    virtual T sum(T value) const = 0;
#define INTERFACE_GATHER(T) virtual std::vector<T> gather(T value, int root) const = 0;

#define WRAP_REDUCE(T) \
    T min(T value) const override { return wrapped.min(value); }\
    T max(T value) const override { return wrapped.max(value); }\
    T sum(T value) const override { return wrapped.sum(value); }
#define WRAP_GATHER(T) std::vector<T> gather(T value, int root) const override { return wrapped.gather(value, root); }

class global_context {
public:
    using spike_vector = std::vector<arb::spike>;

    // default constructor uses a serial context
    global_context(): global_context(serial_context()) {}

    template <typename Impl>
    global_context(Impl&& impl):
        impl_(new wrap<Impl>(std::forward<Impl>(impl)))
    {}

    global_context(global_context&& other) = default;
    global_context& operator=(global_context&& other) = default;

    // better define spikes ahead of time
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


    PUBLIC_REDUCE(float)
    PUBLIC_GATHER(float)
    PUBLIC_REDUCE(double)
    PUBLIC_GATHER(double)
    PUBLIC_REDUCE(int)
    PUBLIC_GATHER(int)
    PUBLIC_REDUCE(std::uint32_t)
    PUBLIC_GATHER(std::uint32_t)
    PUBLIC_REDUCE(std::uint64_t)
    PUBLIC_GATHER(std::uint64_t)
    PUBLIC_GATHER(std::string)

    private:

    struct interface {
        virtual gathered_vector<arb::spike>
            gather_spikes(const spike_vector& local_spikes) const = 0;
        virtual int id() const = 0;
        virtual int size() const = 0;
        virtual void barrier() const = 0;

        INTERFACE_REDUCE(float)
        INTERFACE_GATHER(float)
        INTERFACE_REDUCE(double)
        INTERFACE_GATHER(double)
        INTERFACE_REDUCE(int)
        INTERFACE_GATHER(int)
        INTERFACE_REDUCE(std::uint32_t)
        INTERFACE_GATHER(std::uint32_t)
        INTERFACE_REDUCE(std::uint64_t)
        INTERFACE_GATHER(std::uint64_t)
        INTERFACE_GATHER(std::string)

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

        WRAP_REDUCE(float)
        WRAP_GATHER(float);
        WRAP_REDUCE(double)
        WRAP_GATHER(double);
        WRAP_REDUCE(int)
        WRAP_GATHER(int);
        WRAP_REDUCE(std::uint32_t)
        WRAP_GATHER(std::uint32_t)
        WRAP_REDUCE(std::uint64_t)
        WRAP_GATHER(std::uint64_t)
        WRAP_GATHER(std::string)

        Impl wrapped;
    };

    std::unique_ptr<interface> impl_;
};

} // namespace arb

