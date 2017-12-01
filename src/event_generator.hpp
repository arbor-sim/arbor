#pragma once

#include <cstdint>
#include <memory>
#include <random>

#include <common_types.hpp>
#include <event_queue.hpp>
#include <util/range.hpp>
#include <util/rangeutil.hpp>

namespace arb {

// forward declaration of types required to form event ranges
namespace impl {
    struct pse_iterator;
}
using event_range = util::range<impl::pse_iterator, impl::pse_iterator>;

struct event_generator {
    using iterator = impl::pse_iterator;

    // Return the next event
    // Should return the same event if called multiple times without calling
    // event_generator::pop().
    virtual postsynaptic_spike_event next() = 0;
    virtual void pop() = 0;
    // return all events in half open range [t0, t1)
    virtual event_range events(time_type t0, time_type t1) = 0;
    virtual void reset() = 0;
    virtual ~event_generator() {};
    virtual void advance(time_type t) = 0;
};

namespace impl {

// models input iterator
struct pse_iterator {
    using difference_type = std::ptrdiff_t;
    using value_type = postsynaptic_spike_event;
    using pointer    = value_type*;
    using reference  = value_type&;
    using iterator_category = std::input_iterator_tag;

    struct proxy {
        proxy(postsynaptic_spike_event& ev): e(ev) {}
        postsynaptic_spike_event e;
        postsynaptic_spike_event operator*() {
            return e;
        }
    };

    pse_iterator():
        pse_iterator(time_type(max_time)) // Default to maximum possible time.
    {}

    pse_iterator(event_generator& g):
        gen_(&g), event_(g.next())
    {}

    // sentinel iterator
    // Only time needs to be stored.
    pse_iterator(time_type t):
        gen_(nullptr), event_({{0,0}, t, 0})
    {}

    pse_iterator& operator++() {
        gen_->pop();
        event_ = gen_->next();
        return *this;
    }

    proxy operator++(int) {
        proxy p(event_);
        gen_->pop();
        event_ = gen_->next();
        return p;
    }

    postsynaptic_spike_event operator*() const {
        return event_;
    }

    const postsynaptic_spike_event* operator->() const {
        return &event_;
    }

    bool operator==(const pse_iterator& other) const {
        return other.gen_ ?
            other.event_ == event_:
            other.event_.time<=event_.time;
    }
    bool operator!=(const pse_iterator& other) const {
        return !(*this==other);
    }

    event_generator* gen_;
    postsynaptic_spike_event event_;
};

} // namespace impl

inline
impl::pse_iterator terminal_pse_iterator() {
    return impl::pse_iterator(max_time);
}

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

    event_range events(time_type t0, time_type t1) override {
        advance(t0);
        return {iterator(*this), iterator(t1)};
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

// Generator that feeds events that are specified with a vector
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

    event_range events(time_type t0, time_type t1) override {
        advance(t0);
        return {iterator(*this), iterator(t1)};
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

    event_range events(time_type t0, time_type t1) override {
        advance(t0);
        return {iterator(*this), iterator(t1)};
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
//   - At times described by a Poisson point process with rate 1/dt
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

    event_range events(time_type t0, time_type t1) override {
        advance(t0);
        return {iterator(*this), iterator(t1)};
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

