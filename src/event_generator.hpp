#pragma once

#include <cstdint>
#include <memory>
#include <random>

#include <common_types.hpp>
#include <event_queue.hpp>
#include <util/range.hpp>
#include <util/rangeutil.hpp>

namespace arb {

struct event_generator {
    // Return the next event
    // Should return the same event if called multiple times without calling
    // event_generator::pop().
    virtual postsynaptic_spike_event next() = 0;
    virtual void pop() = 0;
    virtual void reset() = 0;
    virtual ~event_generator() {};
    virtual void advance(time_type t) = 0;
};

inline
postsynaptic_spike_event terminal_pse() {
    return postsynaptic_spike_event{cell_member_type{0,0}, max_time, 0};
}

// Generator that feeds events that are specified with a vector
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

// Generator that for events in a generic sequence
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

// Generates a stream of events
//  * with the first event at time t_start
//  * subsequent events at t_start+n*dt, ∀ n ∈ [0, ∞)
//  * with a set target and weight
struct regular_generator: public event_generator {
    using pse = postsynaptic_spike_event;

    regular_generator(time_type tstart, time_type dt, cell_member_type target, float weight):
        t_start_(tstart), step_(0), dt_(dt),
        target_(target), weight_(weight)
    {}

    postsynaptic_spike_event next() override {
        return {target_, t_start_+(step_*dt_), weight_};
    }

    void pop() override {
        ++step_;
    }

    void advance(time_type t0) override {
        t0 = t0<t_start_? t_start_: t0;
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

    time_type t_start_;
    std::size_t step_;
    time_type dt_;
    cell_member_type target_;
    float weight_;
};

// Generates a stream of events
//  * At times described by a Poisson point process with rate 1/dt
template <typename RandomNumberEngine>
struct poisson_generator: public event_generator {
    using pse = postsynaptic_spike_event;

    poisson_generator(time_type tstart,
                      time_type dt,
                      cell_member_type target,
                      float weight,
                      RandomNumberEngine rng):
        t_start_(tstart),
        exp_(1./dt),
        reset_state_(std::move(rng)),
        target_(target),
        weight_(weight)
    {
        reset();
    }

    postsynaptic_spike_event next() override {
        return {target_, next_, weight_};
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
    const time_type t_start_;
    std::exponential_distribution<time_type> exp_;
    RandomNumberEngine rng_;
    const RandomNumberEngine reset_state_;
    time_type next_;

    const cell_member_type target_;
    const float weight_;
};

using event_generator_ptr = std::unique_ptr<event_generator>;

template <typename T, typename... Args>
event_generator_ptr make_event_generator(Args&&... args) {
    return event_generator_ptr(new T(std::forward<Args>(args)...));
}

} // namespace arb

