#pragma once

#include <ostream>
#include <queue>

#include <cstdint>

namespace nest {
namespace mc {

struct local_event {
    uint32_t target;
    float time;
    float weight;
};

inline bool operator < (local_event const& l, local_event const& r) {
    return l.time < r.time;
}

inline bool operator > (local_event const& l, local_event const& r) {
    return l.time > r.time;
}

class event_queue {
public :
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
    void push(local_event e) {
         queue_.push(e);
    }

    std::size_t size() const {
        return queue_.size();
    }

    // pop until
    std::pair<bool, local_event> pop_if_before(float t_until) {
         if (!queue_.empty() && queue_.top().time < t_until) {
             auto ev = queue_.top();
             queue_.pop();
             return {true, ev};
         }
         else {
             return {false, {}};
         }
    }

private:
    std::priority_queue<
        local_event,
        std::vector<local_event>,
        std::greater<local_event>
    > queue_;
};

} // namespace nest
} // namespace mc

inline
std::ostream& operator<< (std::ostream& o, nest::mc::local_event e)
{
    return o << "event[" << e.target << "," << e.time << "," << e.weight << "]";
}
