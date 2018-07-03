#pragma once

#include <algorithm>
#include <memory>
#include <random>
#include <type_traits>

#include <arbor/common_types.hpp>

#include "util/rangeutil.hpp"

namespace arb {


struct empty_time_seq {
    time_type front() { return terminal_time; }
    void pop() {}
    void reset() {}
    void advance(time_type t) {};
};

// An time_sequence generates a sequence of time points.
// The sequence of times is always monotonically increasing, i.e. each time is
// not earlier than the time that proceded it.
class time_seq {
public:
    //
    // copy, move and constructor interface
    //

    time_seq(): time_seq(empty_time_seq()) {}

    template <
        typename Impl,
        typename = typename std::enable_if<
            !std::is_same<typename std::decay<Impl>::type,
                          time_seq>::value>::type
    >
    time_seq(Impl&& impl):
        impl_(new wrap<Impl>(std::forward<Impl>(impl)))
    {}

    time_seq(time_seq&& other) = default;
    time_seq& operator=(time_seq&& other) = default;

    time_seq(const time_seq& other):
        impl_(other.impl_->clone())
    {}

    time_seq& operator=(const time_seq& other) {
        impl_ = other.impl_->clone();
        return *this;
    }

    //
    // time sequence interface
    //

    // Get the current time in the stream.
    // Does not modify the state of the stream, i.e. multiple calls to
    // front() will return the same time in the absence of calls to pop(),
    // advance() or reset().
    time_type front() {
        return impl_->front();
    }

    // Move the generator to the front time in the stream.
    void pop() {
        impl_->pop();
    }

    // Reset the generator to the same state that it had on construction.
    void reset() {
        impl_->reset();
    }

    // Update state of the generator such that the time returned by front() is
    // the first time with delivery time >= t.
    void advance(time_type t) {
        return impl_->advance(t);
    }

private:
    struct interface {
        virtual time_type front() = 0;
        virtual void pop() = 0;
        virtual void advance(time_type t) = 0;
        virtual void reset() = 0;
        virtual std::unique_ptr<interface> clone() = 0;
        virtual ~interface() {}
    };

    std::unique_ptr<interface> impl_;

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}

        time_type front() override {
            return wrapped.front();
        }
        void pop() override {
            return wrapped.pop();
        }
        void advance(time_type t) override {
            return wrapped.advance(t);
        }
        void reset() override {
            wrapped.reset();
        }
        std::unique_ptr<interface> clone() override {
            return std::unique_ptr<interface>(new wrap<Impl>(wrapped));
        }

        Impl wrapped;
    };
};

// Sequence of time points prescribed by a vector
struct vector_time_seq {
    vector_time_seq(std::vector<time_type> seq):
        seq_(std::move(seq))
    {
        // Ensure that the time values are sorted.
        if (!std::is_sorted(seq_.begin(), seq_.end())) {
            std::sort(seq_.begin(), seq_.end());
        }
        reset();
    }

    time_type front() {
        return it_==seq_.end()? terminal_time: *it_;
    }

    void pop() {
        if (it_!=seq_.end()) {
            ++it_;
        }
    }

    void advance(time_type t0) {
        it_ = std::lower_bound(seq_.begin(), seq_.end(), t0);
    }

    void reset() {
        it_ = seq_.begin();
    }

private:
    std::vector<time_type> seq_;
    std::vector<time_type>::const_iterator it_;
};

// Generates a set of regularly spaced time samples.
//      t=t_start+n*dt, ∀ t ∈ [t_start, t_stop)
struct regular_time_seq {
    regular_time_seq(time_type tstart,
                     time_type dt,
                     time_type tstop=terminal_time):
        step_(0),
        t_start_(tstart),
        dt_(dt),
        t_stop_(tstop)
    {}

    time_type front() {
        const auto t = time();
        return t<t_stop_? t: terminal_time;
    }

    void pop() {
        ++step_;
    }

    void advance(time_type t0) {
        t0 = std::max(t0, t_start_);
        step_ = (t0-t_start_)/dt_;

        // Finding the smallest value for step_ that satisfies the condition
        // that time() >= t0 is unfortunately a horror show because floating
        // point precission.
        while (step_ && time()>=t0) {
            --step_;
        }
        while (time()<t0) {
            ++step_;
        }
    }

    void reset() {
        step_ = 0;
    }

private:
    time_type time() const {
        return t_start_ + step_*dt_;
    }

    std::size_t step_;
    time_type t_start_;
    time_type dt_;
    time_type t_stop_;
};

// Generates a stream of time points described by a Poisson point process
// with rate_per_ms samples per ms.
template <typename RandomNumberEngine>
struct poisson_time_seq {
    poisson_time_seq(RandomNumberEngine rng,
                     time_type tstart,
                     time_type rate_per_ms,
                     time_type tstop=terminal_time):
        exp_(rate_per_ms),
        reset_state_(std::move(rng)),
        t_start_(tstart),
        t_stop_(tstop)
    {
        reset();
    }

    time_type front() {
        return next_<t_stop_? next_: terminal_time;
    }

    void pop() {
        next_ += exp_(rng_);
    }

    void advance(time_type t0) {
        while (next_<t0) {
            pop();
        }
    }

    void reset() {
        rng_ = reset_state_;
        next_ = t_start_;
        pop();
    }

private:
    std::exponential_distribution<time_type> exp_;
    RandomNumberEngine rng_;
    const RandomNumberEngine reset_state_;
    const time_type t_start_;
    const time_type t_stop_;
    time_type next_;
};

} // namespace arb

