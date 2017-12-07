#pragma once

#include <algorithm>
#include <vector>

#include <event_generator.hpp>
#include <event_queue.hpp>
#include <profiling/profiler.hpp>

namespace arb {

// Merge events that are to be delivered from two lists into a sorted list.
// Events are sorted by delivery time, then target, then weight.
// TODO: update this comment.
//
//  tfinal: The time at which the current epoch finishes. The output list, `lc`,
//          will contain all events with delivery times equal to or greater than
//          tfinal.
//  lc: Sorted set of events to be delivered before and after `tfinal`.
//  events: Unsorted list of events with delivery time greater than or equal to
//      tfinal. May be modified by the call.
//  lf: Will hold a list of all postsynaptic events in `events` and `lc` that
//      have delivery times greater than or equal to `tfinal`.
void merge_events(time_type t0,
                  time_type t1,
                  const pse_vector& lc,
                  pse_vector& events,
                  std::vector<event_generator_ptr>& generators,
                  pse_vector& lf);

namespace impl {
    // The tournament tree is used internally by the merge_events method, and
    // it is not intended for use elsewhere. It is exposed here for unit testing
    // of its functionality.
    class tourney_tree {
        using key_val = std::pair<unsigned, postsynaptic_spike_event>;

    public:
        tourney_tree(std::vector<event_generator_ptr>& input);
        bool empty() const;
        bool empty(time_type t) const;
        postsynaptic_spike_event head() const;
        void pop();
        void print() const;

    private:
        void setup(unsigned i);
        void merge_up(unsigned i);
        void update_lane(unsigned lane);
        unsigned parent(unsigned i) const;
        unsigned left(unsigned i) const;
        unsigned right(unsigned i) const;
        unsigned leaf(unsigned i) const;
        bool is_leaf(unsigned i) const;
        const unsigned& id(unsigned i) const;
        postsynaptic_spike_event& event(unsigned i);
        const postsynaptic_spike_event& event(unsigned i) const;
        unsigned next_power_2(unsigned x) const;

        std::vector<key_val> heap_;
        const std::vector<event_generator_ptr>& input_;
        unsigned leaves_;
        unsigned nodes_;
        unsigned n_lanes_;
    };
}

} // namespace arb
