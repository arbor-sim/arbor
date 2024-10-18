#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <queue>
#include <type_traits>
#include <utility>

#include <arbor/common_types.hpp>
#include <arbor/spike_event.hpp>

namespace arb {

/* Event classes `Event` used with `event_queue` must be move and copy constructible,
 * and either have a public field `time` that returns the time value, or provide a
 * projection to a time value through the EventTime functor.
 *
 * Time values must be well ordered with respect to `operator>`.
 */

struct default_event_time {
    template<typename Event>
    auto operator()(Event const& e) const noexcept { return e.time; }
};

template <typename Event, typename EventTime = default_event_time>
class event_queue {
public:
    static constexpr EventTime event_time = {};
    using value_type = Event;
    using event_time_type = decltype(event_time(std::declval<Event>()));

    event_queue() = default;

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
    std::optional<event_time_type> time_if_before(const event_time_type& t_until) {
        if (queue_.empty()) {
            return std::nullopt;
        }
        auto t = event_time(queue_.top());
        return t_until > t? std::optional(t): std::nullopt;
    }

    // Generic conditional pop: pop and return head of queue if
    // queue non-empty and the head satisfies predicate.
    template <typename Pred>
    std::optional<value_type> pop_if(Pred&& pred) {
        if (!queue_.empty() && pred(queue_.top())) {
            auto ev = queue_.top();
            queue_.pop();
            return ev;
        }
        else {
            return std::nullopt;
        }
    }

    // Pop and return top event `ev` of queue if `t_until` > `event_time(ev)`.
    std::optional<value_type> pop_if_before(const event_time_type& t_until) {
        return pop_if(
            [&t_until](const value_type& ev) { return t_until > event_time(ev); }
        );
    }

    // Pop and return top event `ev` of queue unless `event_time(ev)` > `t_until`
    std::optional<value_type> pop_if_not_after(const event_time_type& t_until) {
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
