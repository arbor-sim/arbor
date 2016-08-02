#pragma once

#include <cstdint>
#include <ostream>
#include <queue>

#include "catypes.hpp"
#include "util/optional.hpp"

namespace nest {
namespace mc {

/* An event class Event must comply with the following conventions:
 * Typedefs:
 *     time_type               floating point type used to represent event times
 * Member functions:
 *     time_type when() const  return time value associated with event
 */

template <typename Time>
struct postsynaptic_spike_event {
    using time_type = Time;

    cell_member_type target;
    time_type time;
    float weight;

    time_type when() const { return time; }
};

template <typename Time>
struct sample_event {
    using time_type = Time;

    std::uint32_t sampler_index;
    time_type time;

    time_type when() const { return time; }
};

/* Event objects must have a method event_time() which returns a value
 * from a type with a total ordering with respect to <, >, etc.
 */

template <typename Event>
class event_queue {
public :
    using value_type = Event;
    using time_type = typename Event::time_type;

    // create
    event_queue() {}

    // push stuff
    template <typename Iter>
    void push(Iter b, Iter e) {
         for (; b!=e; ++b) {
             queue_.push(*b);
         }
    }

    // push thing
    void push(const value_type& e) {
         queue_.push(e);
    }

    std::size_t size() const {
        return queue_.size();
    }

    // pop until
    util::optional<value_type> pop_if_before(time_type t_until) {
         if (!queue_.empty() && queue_.top().when() < t_until) {
             auto ev = queue_.top();
             queue_.pop();
             return ev;
         }
         else {
             return util::nothing;
         }
    }

    // clear everything
    void clear() {
        queue_ = decltype(queue_){};
    }

private:
    struct event_greater {
        bool operator()(const Event& a, const Event& b) {
            return a.when() > b.when();
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
