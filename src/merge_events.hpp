#pragma once

#include <algorithm>
#include <vector>

#include <event_generator.hpp>
#include <event_queue.hpp>
#include <profiling/profiler.hpp>

namespace arb {

// merge_events generates a sorted list of postsynaptic events that are to be
// delivered after the current epoch ends. It merges events from multiple
// sources:
//  lc : the list of currently enqueued events
//  pending_events : an unsorted list of events from the communicator
//  generators : a set of event_generators
//
// The time intervales are illustrated below, along with the range of times
// for events in each of lc, events and generators
//  * t the start of the current epoch (not required to perform the merge).
//  * t₀ the start of the next epoch
//  * t₁ the end of the next epoch
//
//   t      t₀     t₁
//   |------|------|
//
//   [----------------------] lc
//          [---------------] pending_events
//          [------) generators
//
// The output list, stored in lf, will contain all the following:
//  * all events in pending_events
//  * events in lc with time >= t₀
//  * events from each generator with time < t₁
// All events in lc that are to be delivered before t₀ are discared, along with
// events from generators after t₁. The generators are left in a state where
// the next event in the generator is the first event with deliver time >= t₁.
void merge_events(time_type t0,
                  time_type t1,
                  const pse_vector& lc,
                  pse_vector& pending_events,
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
