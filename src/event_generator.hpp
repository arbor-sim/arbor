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
struct pse_iterator_sentinel {
    pse_iterator_sentinel(float t): time(t) {}
    float time;
};

// output iterator
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

struct vector_backed_generator: public event_generator {
    vector_backed_generator(std::vector<postsynaptic_spike_event> events):
        events_(std::move(events)),
        it_(events_.begin())
    {}

    postsynaptic_spike_event next() override {
        return *it_;
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

} // namespace arb

