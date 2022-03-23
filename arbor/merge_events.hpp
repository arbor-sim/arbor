#pragma once

#include <iosfwd>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/spike_event.hpp>

#include "profile/profiler_macro.hpp"
#include "util/range.hpp"

// Merge a collection of sorted event sequences into a sorted output sequence.

namespace arb {

using event_span = util::range<const spike_event*>;

void tree_merge_events(std::vector<event_span>& sources, pse_vector& out);

namespace impl {
    // The tournament tree is used internally by the merge_events method, and
    // it is not intended for use elsewhere. It is exposed here for unit testing
    // of its functionality.
    class ARB_ARBOR_API tourney_tree {
        using key_val = std::pair<unsigned, spike_event>;

    public:
        tourney_tree(std::vector<event_span>& input);
        bool empty() const;
        spike_event head() const;
        void pop();
        friend std::ostream& operator<<(std::ostream&, const tourney_tree&);

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
        spike_event& event(unsigned i);
        const spike_event& event(unsigned i) const;
        unsigned next_power_2(unsigned x) const;

        std::vector<key_val> heap_;
        std::vector<event_span>& input_;
        unsigned leaves_;
        unsigned nodes_;
        unsigned n_lanes_;
    };
}

} // namespace arb
