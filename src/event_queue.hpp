#pragma once

#include <cstdint>
#include <limits>
#include <ostream>
#include <queue>
#include <type_traits>
#include <utility>

#include "common_types.hpp"
#include "util/meta.hpp"
#include "util/optional.hpp"
#include "util/range.hpp"
#include "util/strprintf.hpp"

namespace nest {
namespace mc {

/* Event classes `Event` used with `event_queue` must be move and copy constructible,
 * and either have a public field `time` that returns the time value, or provide an
 * overload of `event_time(const Event&)` which returns this value.
 *
 * Time values must be well ordered with respect to `operator>`.
 */

struct postsynaptic_spike_event {
    cell_member_type target;
    time_type time;
    float weight;
};

struct sample_event {
    using size_type = std::uint32_t;

    size_type sampler_index;
    time_type time;
};

// Configuration point: define `event_time(ev)` for event objects `ev`
// that do not have the corresponding `time` member field.

template <typename Event>
auto event_time(const Event& ev) -> decltype(ev.time) {
    return ev.time;
}

namespace impl {
    using ::nest::mc::event_time;

    // wrap in `impl::` namespace to obtain correct ADL for return type.
    template <typename Event>
    using event_time_type = decltype(event_time(std::declval<Event>()));
}

template <typename Event>
class event_queue {
public :
    using value_type = Event;
    using event_time_type = impl::event_time_type<Event>;

    event_queue() {}

    void push(const value_type& e) {
         queue_.push(e);
    }

    void push(value_type&& e) {
         queue_.push(std::move(e));
    }

    bool empty() const {
        return size()==0;
    }

    std::size_t size() const {
        return queue_.size();
    }

    // Return time t of head of queue if `t_until` > `t`.
    util::optional<event_time_type> time_if_before(const event_time_type& t_until) {
        if (queue_.empty()) {
            return util::nothing;
        }

        using ::nest::mc::event_time;
        auto t = event_time(queue_.top());
        return t_until > t? util::just(t): util::nothing;
    }

    // Generic conditional pop: pop and return head of queue if
    // queue non-empty and the head satisfies predicate.
    template <typename Pred>
    util::optional<value_type> pop_if(Pred&& pred) {
        using ::nest::mc::event_time;
        if (!queue_.empty() && pred(queue_.top())) {
            auto ev = queue_.top();
            queue_.pop();
            return ev;
        }
        else {
            return util::nothing;
        }
    }

    // Pop and return top event `ev` of queue if `t_until` > `event_time(ev)`.
    util::optional<value_type> pop_if_before(const event_time_type& t_until) {
        using ::nest::mc::event_time;
        return pop_if(
            [&t_until](const value_type& ev) { return t_until > event_time(ev); }
        );
    }

    // Pop and return top event `ev` of queue unless `event_time(ev)` > `t_until`
    util::optional<value_type> pop_if_not_after(const event_time_type& t_until) {
        using ::nest::mc::event_time;
        return pop_if(
            [&t_until](const value_type& ev) { return !(event_time(ev) > t_until); }
        );
    }

    // Clear queue and free storage.
    void clear() {
        queue_ = decltype(queue_){};
    }

private:
    struct event_greater {
        bool operator()(const Event& a, const Event& b) {
            using ::nest::mc::event_time;
            return event_time(a) > event_time(b);
        }
    };

    std::priority_queue<
        Event,
        std::vector<Event>,
        event_greater
    > queue_;
};

// Indexed collection of pop-only queues.
// 'flat' implementation below is acting as a prototype for GPU back-end.

template <typename Event>
class multi_event_stream {
public:
    using value_type = Event;
    using time_type = impl::event_time_type<Event>;
    using size_type = unsigned;

    multi_event_stream() {}

    explicit multi_event_stream(size_type n_stream):
       span_(n_stream, {0u, 0u}), mark_(n_stream, 0u) {}

    size_type n_streams() const { return span_.size(); }
    size_type size() const { return n_streams(); }

    size_type n_events(size_type i) const {
        auto span = span_[i];
        return span.second-span.first;
    }

    size_type remaining() const {
        return remaining_;
    }

    void clear() {
        ev_.clear();
        remaining_ = 0;
        span_.assign(n_streams(), {0u, 0u});
    }

    // Load events from a sequence (of length at least `size()`) of
    // sequences of events.
    template <typename EvSeqs>
    void init(const EvSeqs& events) {
        using std::begin;
        using std::end;

        EXPECTS(n_streams()==util::size(events));

        remaining_ = 0;
        auto evi = begin(events);
        for (size_type i = 0; i<n_streams(); ++i) {
            if (evi != end(events)) {
                span_[i].first = size_type(ev_.size());
                ev_.insert(ev_.end(), begin(*evi), end(*evi));
                span_[i].second = size_type(ev_.size());
                mark_[i] = span_[i].first;
                ++evi;

                // check size for wrapping!
                if (ev_.size()>std::numeric_limits<size_type>::max()) {
                    throw std::range_error("too many events");
                }
            }
            else {
                span_[i] = {0u, 0u};
            }
        }
        remaining_ = ev_.size();
    }

    // Designate for processing events `ev` at head of `i`th event stream
    // until `event_time(ev)` > `t_until`.
    void mark_until_after(size_type i, time_type t_until) {
        using ::nest::mc::event_time;

        auto& span = span_[i];
        auto& mark = mark_[i];

        mark = span.first;
        while (mark!=span.second && !(event_time(ev_[mark])>t_until)) {
            ++mark;
        }
    }

    // Remove marked events `ev` from front of `i`th event stream.
    void drop_marked_events(size_type i) {
        remaining_ -= (mark_[i]-span_[i].first);
        span_[i].first = mark_[i];
    }

    // Return range of marked events.
    util::range<value_type*> marked_events(size_type i) {
        return {&ev_[span_[i].first], &ev_[mark_[i]]};
    }

    // Return time of head of `i`th event stream if less than `t_until`, or else `t_until`.
    time_type event_time_if_before(size_type i, const time_type& t_until) {
        if (!n_events(i)) {
            return t_until;
        }

        using ::nest::mc::event_time;
        auto t = event_time(ev_[span_[i].first]);
        return t_until > t? t: t_until;
    }

    // TODO: remove once we are confident implementation of lowered cell event delivery
    // is sound.
    friend std::ostream& operator<<(std::ostream& out, const multi_event_stream& m) {
        using ::nest::mc::event_time;

        auto n = m.n_streams();

        out << "[";
        unsigned i = 0;
        for (unsigned ev_i = 0; ev_i<m.ev_.size(); ++ev_i) {
            while (m.span_[i].second<=ev_i && i<n) ++i;
            if (i<n) {
                out << util::strprintf(" % 6d ", i);
            }
            else {
                out << "      ?";
            }
        }
        out << "\n[";

        i = 0;
        for (unsigned ev_i = 0; ev_i<m.ev_.size(); ++ev_i) {
            while (m.span_[i].second<=ev_i && i<n) ++i;

            bool discarded = i<n && m.span_[i].first>ev_i;
            bool marked = i<n && m.mark_[i].first>ev_i;

            if (discarded) {
                out << "       x";
            }
            else {
                out << util::strprintf(" % 6.3f%c", event_time(m.ev_[i]), marked?'*':' ');
            }
        }
        out << "]\n";
        return out;
    }

private:
    const value_type& top_unsafe(size_type i) const {
        return ev_[span_[i].first];
    }

    std::vector<value_type> ev_;
    std::vector<std::pair<size_type, size_type>> span_;
    std::vector<size_type> mark_;
    size_type remaining_ = 0;
};

} // namespace nest
} // namespace mc

inline std::ostream& operator<<(
    std::ostream& o, const nest::mc::postsynaptic_spike_event& e)
{
    return o << "event[" << e.target << "," << e.time << "," << e.weight << "]";
}
