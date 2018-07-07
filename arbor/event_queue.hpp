#pragma once

#include <cstdint>
#include <limits>
#include <ostream>
#include <queue>
#include <type_traits>
#include <utility>

#include <arbor/common_types.hpp>
#include <arbor/spike_event.hpp>
#include <arbor/util/optional.hpp>
#include <arbor/generic_event.hpp>

namespace arb {

/* Event classes `Event` used with `event_queue` must be move and copy constructible,
 * and either have a public field `time` that returns the time value, or provide an
 * overload of `event_time(const Event&)` which returns this value (see generic_event.hpp).
 *
 * Time values must be well ordered with respect to `operator>`.
 */

template <typename Event>
class event_queue {
public:
    using value_type = Event;
    using event_time_type = ::arb::event_time_type<Event>;

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
            return util::nullopt;
        }

        using ::arb::event_time;
        auto t = event_time(queue_.top());
        return t_until > t? util::just(t): util::nullopt;
    }

    // Generic conditional pop: pop and return head of queue if
    // queue non-empty and the head satisfies predicate.
    template <typename Pred>
    util::optional<value_type> pop_if(Pred&& pred) {
        using ::arb::event_time;
        if (!queue_.empty() && pred(queue_.top())) {
            auto ev = queue_.top();
            queue_.pop();
            return ev;
        }
        else {
            return util::nullopt;
        }
    }

    // Pop and return top event `ev` of queue if `t_until` > `event_time(ev)`.
    util::optional<value_type> pop_if_before(const event_time_type& t_until) {
        using ::arb::event_time;
        return pop_if(
            [&t_until](const value_type& ev) { return t_until > event_time(ev); }
        );
    }

    // Pop and return top event `ev` of queue unless `event_time(ev)` > `t_until`
    util::optional<value_type> pop_if_not_after(const event_time_type& t_until) {
        using ::arb::event_time;
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
            using ::arb::event_time;
            return event_time(a) > event_time(b);
        }
    };

    std::priority_queue<
        Event,
        std::vector<Event>,
        event_greater
    > queue_;
};

} // namespace arb
