#pragma once

#include <arbor/export.hpp>
#include <arbor/spike_event.hpp>

#include "util/range.hpp"

namespace arb {

using event_span = util::range<const spike_event*>;

// The tournament tree is used internally by the merge_events method, and
// it is not intended for use elsewhere. It is exposed here for unit testing
// of its functionality.
class ARB_ARBOR_API tourney_tree final {

    using key = unsigned;
    using val = spike_event;

public:
    tourney_tree(std::vector<event_span>&& input);
    bool empty() const;
    spike_event head() const;
    spike_event pop();
    friend std::ostream& operator<<(std::ostream&, const tourney_tree&);
    std::size_t size() const;
private:
    inline void setup(unsigned i);
    inline void merge_up(unsigned i);
    inline void update_lane(unsigned lane);
    inline unsigned parent(unsigned i) const;
    inline unsigned left(unsigned i) const;
    inline unsigned right(unsigned i) const;
    inline unsigned leaf(unsigned i) const;
    inline bool is_leaf(unsigned i) const;
    inline const unsigned& id(unsigned i) const;
    inline spike_event& event(unsigned i);
    inline const spike_event& event(unsigned i) const;
    inline unsigned next_power_2(unsigned x) const;

    std::vector<key> heap_keys_;
    std::vector<val> heap_vals_;
    std::vector<event_span>& input_;
    unsigned n_lanes_;
    unsigned leaves_;
    unsigned nodes_;
};

void tree_merge_events(std::vector<event_span> sources, pse_vector& out);

}
