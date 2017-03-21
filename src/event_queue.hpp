#pragma once

#include <cstdint>
#include <ostream>
#include <queue>
#include <type_traits>

#include "common_types.hpp"
#include "util/meta.hpp"
#include "util/optional.hpp"

namespace nest {
namespace mc {

/* Event classes `Event` used with `event_queue` must be move and copy constructible,
 * and either have a public field `time` that returns the time value, or provide an
 * overload of `event_time(const Event&)` which returns this value.
 *
 * Time values must be well ordered with respect to `operator>`.
 */

template <typename Time>
struct postsynaptic_spike_event {
    using time_type = Time;

    cell_member_type target;
    time_type time;
    float weight;
};

template <typename Time>
struct sample_event {
    using time_type = Time;

    std::uint32_t sampler_index;
    time_type time;
};

namespace impl {
    template <typename X>
    struct has_time_field {
        template <typename T>
        static std::false_type test(...) {}
        template <typename T>
        static decltype(std::declval<T>().time, std::true_type{}) test(int) {}

        using type = decltype(test<X>(0));
        static constexpr bool value = type::value;
    };
}

// Configuration point: define `event_time(ev)` for event objects `ev`
// that do not have the corresponding `time` member field.

template <typename Event, typename = util::enable_if_t<impl::has_time_field<Event>::value>>
auto event_time(const Event& ev) -> decltype(ev.time) {
    return ev.time;
}

namespace impl {
    // use `impl::` version to obtain correct ADL for return type.
    using ::nest::mc::event_time;

    template <typename Event>
    using event_time_type = decltype(event_time(std::declval<Event>()));
}

template <typename Event>
class event_queue {
public :
    using value_type = Event;
    using time_type = impl::event_time_type<Event>;

    event_queue() {}

    void push(const value_type& e) {
         queue_.push(e);
    }

    bool empty() const {
        return size()==0;
    }

    std::size_t size() const {
        return queue_.size();
    }

    // Pop and return top event `ev` of queue if `t_until` > `event_time(ev)`.
    util::optional<value_type> pop_if_before(const time_type& t_until) {
        using ::nest::mc::event_time;
        if (!queue_.empty() && t_until > event_time(queue_.top())) {
            auto ev = queue_.top();
            queue_.pop();
            return ev;
        }
        else {
            return util::nothing;
        }
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

} // namespace nest
} // namespace mc

template <typename T>
inline std::ostream& operator<<(std::ostream& o, const nest::mc::postsynaptic_spike_event<T>& e)
{
    return o << "event[" << e.target << "," << e.time << "," << e.weight << "]";
}
