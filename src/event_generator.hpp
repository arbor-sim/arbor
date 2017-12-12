#pragma once

#include <cstdint>
#include <memory>
#include <random>

#include <common_types.hpp>
#include <event_queue.hpp>
#include <util/range.hpp>
#include <util/rangeutil.hpp>

namespace arb {

// An event_generator generates a sequence of events to be delivered to a cell.
// The sequence of events is always in ascending order, i.e. each event will be
// greater than the event that proceded it, where events are ordered by:
//  - delivery time;
//  - then target id for events with the same delivery time;
//  - then weight for events with the same delivery time and target.
struct event_generator {
    // Return the next event in the stream.
    // Returns the same event if called multiple times without calling pop().
    virtual postsynaptic_spike_event next() = 0;

    // Move the generator to the next event in the stream.
    virtual void pop() = 0;

    // Reset the generator to the same state that it had on construction.
    virtual void reset() = 0;

    // Update state of the generator such that the event returned by next() is
    // the first event with delivery time >= t.
    virtual void advance(time_type t) = 0;

    virtual ~event_generator() {};
};

inline
postsynaptic_spike_event terminal_pse() {
    return postsynaptic_spike_event{cell_member_type{0,0}, max_time, 0};
}

// Generator that feeds events that are specified with a vector.
// Makes a copy of the input sequence of events.
struct vector_backed_generator: public event_generator {
    using pse = postsynaptic_spike_event;
    vector_backed_generator(pse_vector events):
        events_(std::move(events)),
        it_(events_.begin())
    {
        if (!std::is_sorted(events_.begin(), events_.end())) {
            util::sort(events_);
        }
    }

    postsynaptic_spike_event next() override {
        return it_==events_.end()? terminal_pse(): *it_;
    }

    void pop() override {
        if (it_!=events_.end()) {
            ++it_;
        }
    }

    void reset() override {
        it_ = events_.begin();
    }

    void advance(time_type t) override {
        it_ = std::lower_bound(events_.begin(), events_.end(), t, event_time_less());
    }

private:
    std::vector<postsynaptic_spike_event> events_;
    std::vector<postsynaptic_spike_event>::const_iterator it_;
};

// Generator for events in a generic sequence.
// The generator keeps a reference to a Seq, i.e. it does not own the sequence.
// Care must be taken to avoid lifetime issues, to ensure that the generator
// does not outlive the sequence.
template <typename Seq>
struct seq_generator: public event_generator {
    using pse = postsynaptic_spike_event;
    seq_generator(Seq& events):
        events_(events),
        it_(std::begin(events_))
    {
        EXPECTS(std::is_sorted(events_.begin(), events_.end()));
    }

    postsynaptic_spike_event next() override {
        return it_==events_.end()? terminal_pse(): *it_;
    }

    void pop() override {
        if (it_!=events_.end()) {
            ++it_;
        }
    }

    void reset() override {
        it_ = events_.begin();
    }

    void advance(time_type t) override {
        it_ = std::lower_bound(events_.begin(), events_.end(), t, event_time_less());
    }

private:

    const Seq& events_;
    typename Seq::const_iterator it_;
};

// Generates a set of regularly spaced events:
//  * with delivery times t=t_start+n*dt, ∀ t ∈ [t_start, t_stop)
//  * with a set target and weight
struct regular_generator: public event_generator {
    using pse = postsynaptic_spike_event;

    regular_generator(cell_member_type target,
                      float weight,
                      time_type tstart,
                      time_type dt,
                      time_type tstop=max_time):
        target_(target),
        weight_(weight),
        step_(0),
        t_start_(tstart),
        dt_(dt),
        t_stop_(tstop)
    {}

    postsynaptic_spike_event next() override {
        const auto t = time();
        return t<t_stop_?
            postsynaptic_spike_event{target_, t, weight_}:
            terminal_pse();
    }

    void pop() override {
        ++step_;
    }

    void advance(time_type t0) override {
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

    void reset() override {
        step_ = 0;
    }

private:
    time_type time() const {
        return t_start_ + step_*dt_;
    }

    cell_member_type target_;
    float weight_;
    std::size_t step_;
    time_type t_start_;
    time_type dt_;
    time_type t_stop_;
};

// Generates a stream of events at times described by a Poisson point process
// with rate_per_ms spikes per ms.
template <typename RandomNumberEngine>
struct poisson_generator: public event_generator {
    using pse = postsynaptic_spike_event;

    poisson_generator(cell_member_type target,
                      float weight,
                      RandomNumberEngine rng,
                      time_type tstart,
                      time_type rate_per_ms,
                      time_type tstop=max_time):
        exp_(rate_per_ms),
        reset_state_(std::move(rng)),
        target_(target),
        weight_(weight),
        t_start_(tstart),
        t_stop_(tstop)
    {
        reset();
    }

    postsynaptic_spike_event next() override {
        return next_<t_stop_?
            postsynaptic_spike_event{target_, next_, weight_}:
            terminal_pse();
    }

    void pop() override {
        next_ += exp_(rng_);
    }

    void advance(time_type t0) override {
        while (next_<t0) {
            pop();
        }
    }

    void reset() override {
        rng_ = reset_state_;
        next_ = t_start_;
        pop();
    }

private:
    std::exponential_distribution<time_type> exp_;
    RandomNumberEngine rng_;
    const RandomNumberEngine reset_state_;
    const cell_member_type target_;
    const float weight_;
    const time_type t_start_;
    const time_type t_stop_;
    time_type next_;

};

using event_generator_ptr = std::unique_ptr<event_generator>;

template <typename T, typename... Args>
event_generator_ptr make_event_generator(Args&&... args) {
    return event_generator_ptr(new T(std::forward<Args>(args)...));
}

} // namespace arb

