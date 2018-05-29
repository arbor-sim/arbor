#pragma once

#include <cstdint>
#include <memory>
#include <random>

#include <common_types.hpp>
#include <event_queue.hpp>
#include <time_sequence.hpp>
#include <util/range.hpp>
#include <util/rangeutil.hpp>

namespace arb {

inline
postsynaptic_spike_event terminal_pse() {
    return postsynaptic_spike_event{cell_member_type{0,0}, max_time, 0};
}

inline
bool is_terminal_pse(const postsynaptic_spike_event& e) {
    return e.time==max_time;
}

// An event_generator generates a sequence of events to be delivered to a cell.
// The sequence of events is always in ascending order, i.e. each event will be
// greater than the event that proceded it, where events are ordered by:
//  - delivery time;
//  - then target id for events with the same delivery time;
//  - then weight for events with the same delivery time and target.
class event_generator {
public:
    //
    // copy, move and constructor interface
    //

    event_generator(): event_generator(dummy_generator()) {}

    template <typename Impl>
    event_generator(Impl&& impl):
        impl_(new wrap<Impl>(std::forward<Impl>(impl)))
    {}

    event_generator(event_generator&& other) = default;
    event_generator& operator=(event_generator&& other) = default;

    event_generator(const event_generator& other):
        impl_(other.impl_->clone())
    {}

    event_generator& operator=(const event_generator& other) {
        impl_ = other.impl_->clone();
        return *this;
    }

    //
    // event generator interface
    //

    // Get the current event in the stream.
    // Does not modify the state of the stream, i.e. multiple calls to
    // next() will return the same event in the absence of calls to pop(),
    // advance() or reset().
    postsynaptic_spike_event next() {
        return impl_->next();
    }

    // Move the generator to the next event in the stream.
    void pop() {
        impl_->pop();
    }

    // Reset the generator to the same state that it had on construction.
    void reset() {
        impl_->reset();
    }

    // Update state of the generator such that the event returned by next() is
    // the first event with delivery time >= t.
    void advance(time_type t) {
        return impl_->advance(t);
    }

private:
    struct interface {
        virtual postsynaptic_spike_event next() = 0;
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

        postsynaptic_spike_event next() override {
            return wrapped.next();
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

    struct dummy_generator {
        postsynaptic_spike_event next() { return terminal_pse(); }
        void pop() {}
        void reset() {}
        void advance(time_type t) {};
    };

};

// Generator that feeds events that are specified with a vector.
// Makes a copy of the input sequence of events.
struct vector_backed_generator {
    using pse = postsynaptic_spike_event;
    vector_backed_generator(cell_member_type target, float weight, std::vector<time_type> samples):
        target_(target),
        weight_(weight),
        tseq_(std::move(samples))
    {}

    postsynaptic_spike_event next() {
        return postsynaptic_spike_event{target_, tseq_.next(), weight_};
    }

    void pop() {
        tseq_.pop();
    }

    void reset() {
        tseq_.reset();
    }

    void advance(time_type t) {
        tseq_.advance(t);
    }

private:
    cell_member_type target_;
    float weight_;
    vector_time_seq tseq_;
};

// Generator for events in a generic sequence.
// The generator keeps a reference to a Seq, i.e. it does not own the sequence.
// Care must be taken to avoid lifetime issues, to ensure that the generator
// does not outlive the sequence.
template <typename Seq>
struct seq_generator {
    using pse = postsynaptic_spike_event;
    seq_generator(Seq& events):
        events_(events),
        it_(std::begin(events_))
    {
        EXPECTS(util::is_sorted(events_));
    }

    postsynaptic_spike_event next() {
        return it_==events_.end()? terminal_pse(): *it_;
    }

    void pop() {
        if (it_!=events_.end()) {
            ++it_;
        }
    }

    void reset() {
        it_ = events_.begin();
    }

    void advance(time_type t) {
        it_ = std::lower_bound(events_.begin(), events_.end(), t, event_time_less());
    }

private:

    const Seq& events_;
    typename Seq::const_iterator it_;
};

// Generates a set of regularly spaced events:
//  * with delivery times t=t_start+n*dt, ∀ t ∈ [t_start, t_stop)
//  * with a set target and weight
struct regular_generator {
    using pse = postsynaptic_spike_event;

    regular_generator(cell_member_type target,
                      float weight,
                      time_type tstart,
                      time_type dt,
                      time_type tstop=max_time):
        target_(target),
        weight_(weight),
        tseq_(tstart, dt, tstop)
    {}

    postsynaptic_spike_event next() {
        return postsynaptic_spike_event{target_, tseq_.next(), weight_};
    }

    void pop() {
        tseq_.pop();
    }

    void advance(time_type t0) {
        tseq_.advance(t0);
    }

    void reset() {
        tseq_.reset();
    }

private:

    cell_member_type target_;
    float weight_;
    regular_time_seq tseq_;
};

// Generates a stream of events at times described by a Poisson point process
// with rate_per_ms spikes per ms.
template <typename RandomNumberEngine>
struct poisson_generator {
    using pse = postsynaptic_spike_event;

    poisson_generator(cell_member_type target,
                      float weight,
                      RandomNumberEngine rng,
                      time_type tstart,
                      time_type rate_per_ms,
                      time_type tstop=max_time):
        target_(target),
        weight_(weight),
        tseq_(std::move(rng), tstart, rate_per_ms, tstop)
    {
        reset();
    }

    postsynaptic_spike_event next() {
        return postsynaptic_spike_event{target_, tseq_.next(), weight_};
    }

    void pop() {
        tseq_.pop();
    }

    void advance(time_type t0) {
        tseq_.advance(t0);
    }

    void reset() {
        tseq_.reset();
    }

private:
    const cell_member_type target_;
    const float weight_;
    poisson_time_seq<RandomNumberEngine> tseq_;
};

} // namespace arb

