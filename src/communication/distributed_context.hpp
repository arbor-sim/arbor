#pragma once

#include <string>

#include <spike.hpp>
#include <communication/gathered_vector.hpp>

#if defined(ARB_HAVE_MPI)
#   include "mpi_context.hpp"
#endif
#include "serial_context.hpp"


namespace arb {

#define PUBLIC_REDUCE(T) \
    T min(T value) const { return impl_->min(value); }\
    T max(T value) const { return impl_->max(value); }\
    T sum(T value) const { return impl_->sum(value); }
#define PUBLIC_GATHER(T) std::vector<T> gather(T value, int root) const { return impl_->gather(value, root); }
#define PUBLIC_ALL(T) PUBLIC_REDUCE(T) PUBLIC_GATHER(T)

#define INTERFACE_REDUCE(T) \
    virtual T min(T value) const = 0;\
    virtual T max(T value) const = 0;\
    virtual T sum(T value) const = 0;
#define INTERFACE_GATHER(T) virtual std::vector<T> gather(T value, int root) const = 0;
#define INTERFACE_ALL(T) INTERFACE_REDUCE(T) INTERFACE_GATHER(T)

#define WRAP_REDUCE(T) \
    T min(T value) const override { return wrapped.min(value); }\
    T max(T value) const override { return wrapped.max(value); }\
    T sum(T value) const override { return wrapped.sum(value); }
#define WRAP_GATHER(T) std::vector<T> gather(T value, int root) const override { return wrapped.gather(value, root); }
#define WRAP_ALL(T) WRAP_REDUCE(T) WRAP_GATHER(T)

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


    PUBLIC_ALL(float)
    PUBLIC_ALL(double)
    PUBLIC_ALL(int)
    PUBLIC_ALL(std::uint32_t)
    PUBLIC_ALL(std::uint64_t)
    PUBLIC_GATHER(std::string)

    private:

    struct interface {
        virtual gathered_vector<arb::spike>
            gather_spikes(const spike_vector& local_spikes) const = 0;
        virtual int id() const = 0;
        virtual int size() const = 0;
        virtual void barrier() const = 0;
        virtual std::string name() const = 0;

        INTERFACE_ALL(float)
        INTERFACE_ALL(double)
        INTERFACE_ALL(int)
        INTERFACE_ALL(std::uint32_t)
        INTERFACE_ALL(std::uint64_t)
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
        std::string name() const {
            return wrapped.name();
        }

        WRAP_ALL(float)
        WRAP_ALL(double)
        WRAP_ALL(int)
        WRAP_ALL(std::uint32_t)
        WRAP_ALL(std::uint64_t)
        WRAP_GATHER(std::string)

        Impl wrapped;
    };

    std::unique_ptr<interface> impl_;
};

} // namespace arb

