#pragma once

#include <cstdint>
#include <ostream>
#include <queue>

#include "catypes.hpp"
#include "util/optional.hpp"

namespace nest {
namespace mc {

struct postsynaptic_spike_event {
    //cell_member_type target;
    std::uint32_t target;
    float time;
    float weight;
};

inline float event_time(const postsynaptic_spike_event &ev) { return ev.time; }

struct sample_event {
    std::uint32_t sampler_index;
    float time;
};

inline float event_time(const sample_event &ev) { return ev.time; }

/* Event objects must have a method event_time() which returns a value
 * from a type with a total ordering with respect to <, >, etc.
 */

template <typename Event>
class event_queue {
public :
    using value_type = Event;
    using time_type = decltype(event_time(std::declval<Event>()));

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
    void push(const value_type &e) {
         queue_.push(e);
    }

    std::size_t size() const {
        return queue_.size();
    }

    // pop until
    util::optional<value_type> pop_if_before(time_type t_until) {
         if (!queue_.empty() && event_time(queue_.top()) < t_until) {
             auto ev = queue_.top();
             queue_.pop();
             return ev;
         }
         else {
             return util::nothing;
         }
    }

private:
    struct event_greater {
        bool operator()(const Event &a, const Event &b) {
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

inline
std::ostream& operator<< (std::ostream& o, const nest::mc::postsynaptic_spike_event& e)
{
    return o << "event[" << e.target << "," << e.time << "," << e.weight << "]";
}
