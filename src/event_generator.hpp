#pragma once

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
    virtual postsynaptic_spike_event next() = 0;
    virtual void pop() = 0;
    // return all events in half open range [t0, t1)
    virtual event_range events(time_type t0, time_type t1) = 0;
    virtual void reset() = 0;
    virtual ~event_generator() {};
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

    pse_iterator(event_generator& g): gen_(&g), event_(g.next()) {}

    // sentinel iterator
    // Only time needs to be stored.
    pse_iterator(time_type tfinal): gen_(nullptr) {
        event_.time = tfinal;
    }

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

// Generator that feeds events that are specified with a vector
struct vector_backed_generator: public event_generator {
    using pse = postsynaptic_spike_event;
    vector_backed_generator(pse_vector events):
        events_(std::move(events)),
        it_(events_.begin())
    {}

    postsynaptic_spike_event next() override {
        if (it_!=events_.end()) {
            return *it_;
        }
        else {
            return {{0, 0}, std::numeric_limits<time_type>::infinity(), 0.0};
        }
    }

    void pop() override {
        if (it_!=events_.end()) {
            ++it_;
        }
    }

    event_range events(time_type t0, time_type t1) override {
        advance(t0);
        return {impl::pse_iterator(*this), impl::pse_iterator(t1)};
    }

    void reset() override {
        it_ = events_.begin();
    }

private:

    void advance(time_type t) {
        it_ = std::lower_bound(events_.begin(), events_.end(), t, event_time_less());
    }
    std::vector<postsynaptic_spike_event> events_;
    std::vector<postsynaptic_spike_event>::const_iterator it_;
};

// Generates a stream of events
//  * with the first event at time t_start
//  * subsequent events at t_start+n*dt
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
        t0 = t0<t_start_? t_start_: t0;
        step_ = (t0-t_start_)/dt_;
        while (time()<t0) {
            ++step_;
        }
        return {impl::pse_iterator(*this), impl::pse_iterator(t1)};
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

} // namespace arb

