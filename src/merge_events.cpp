#include <set>
#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <cell_group_factory.hpp>
#include <domain_decomposition.hpp>
#include <merge_events.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <util/filter.hpp>
#include <util/span.hpp>
#include <util/unique_any.hpp>
#include <profiling/profiler.hpp>

namespace arb {

namespace impl {

constexpr inline
unsigned parent(unsigned i) {
    return (i-1)>>1;
}

constexpr inline
unsigned left(unsigned i) {
    return (i<<1) + 1;
}

constexpr inline
unsigned right(unsigned i) {
    return (i<<1) + 2; // left(i)+1
}

// The tournament tree data structure is used to merge k sorted lists of events.
// See online for high-level information about tournament trees.
//
// This implementation maintains a heap-like data structure, with entries of type:
//      std::pair<unsigned, post_synaptic_event>
// where the unsigned ∈ [0, k-1] is the id of the list from which the event was
// drawn. The id is stored so that the operation of removing the most recent event
// knows which leaf node needs to be updated (i.e. the leaf node of the list from
// which the most recent event was drawn).
//
// unsigned is used for storing the index, because if drawing events from more
// event generators than can be counted using an unsigned a complete redesign
// will be needed.
tourney_tree::tourney_tree(std::vector<event_generator>& input):
    input_(input),
    n_lanes_(input_.size())
{
    // Must have at least 3 queues
    if (n_lanes_<=2) {
        throw std::runtime_error(
            "A tourney_tree must be initialized with at least 3 sequences to merge");
    }
    // Maximum value in unsigned limits how many queues we can have
    EXPECTS(n_lanes_<(1u<<(sizeof(unsigned)*8u-1u)));

    leaves_ = next_power_2(n_lanes_);
    nodes_ = leaves_-1;

    // Allocate space for the tree nodes
    index_tree_.resize(nodes_);
    events_.resize(leaves_, terminal_pse());

    // Set the leaf nodes
    for (auto i=0u; i<n_lanes_; ++i) {
        events_[i] = input[i].next();
    }

    // Initialize the index_tree_

    // Handle special case for 4 leaves seperately
    if (leaves_==4) {
        const auto i = events_[0]<events_[1]? 0: 1;
        index_tree_[1] = i;
        const auto j = events_[2]<events_[3]? 2: 3;
        index_tree_[2] = j;
        index_tree_[0] = events_[i]<events_[j]? i: j;
        return;
    }
    // Walk the tree to initialize the non-leaf nodes
    setup(0);
}

void tourney_tree::print() const {
    printf("%6s%8s%8s%8s\n", "lane", "time", "target", "weight");
    unsigned i = 0u;
    for (auto& e: events_) {
        printf("%6u%8.4f%4u.%-3u%8.2f\n", i++, e.time, e.target.gid, e.target.index, e.weight);
    }

    unsigned end = 1;
    i = 0;
    while (end <= nodes_) {
        while (i<end) std::cout << index_tree_[i++] << " ";
        std::cout << "\n";
        end = 2*end+1;
    }
}

bool tourney_tree::empty() const {
    return event(0).time == max_time;
}

bool tourney_tree::empty(time_type t) const {
    return event(0).time >= t;
}

postsynaptic_spike_event tourney_tree::head() const {
    return event(0);
}

// Remove the smallest (most recent) event from the tree, then update the
// tree so that head() returns the next event.
void tourney_tree::pop() {
    unsigned lane = index_tree_[0];
    // draw the next event from the input lane
    input_[lane].pop();
    // place event the leaf node for this lane
    events_[lane] = input_[lane].next();

    // re-heapify the tree with a single walk from leaf to root

    // special cases for 3 and 4 lanes.
    if (n_lanes_==3) {
        if (lane<2) {
            const auto i = events_[0]<events_[1]? 0: 1;
            index_tree_[1] = i;
            index_tree_[0] = events_[i]<events_[2]? i: 2;
            return;
        }
        const auto i = index_tree_[1];
        index_tree_[0] = events_[i]<events_[2]? i: 2;

        // The two lines below are a branch-free alternative. They always require
        // two comparisons, while avoiding one write to index_tree_.
        //
        // const auto i = events_[0]<events_[1]? 0: 1;
        // index_tree_[0] = events_[i]<events_[2]? i: 2;

        return;
    }
    else if (n_lanes_==4) {
        if (lane<2) {
            const auto i = events_[0]<events_[1]? 0: 1;
            index_tree_[1] = i;
            const auto j = index_tree_[2];
            index_tree_[0] = events_[i]<events_[j]? i: j;
            return;
        }
        const auto i = events_[2]<events_[3]? 2: 3;
        index_tree_[2] = i;
        const auto j = index_tree_[1];
        index_tree_[0] = events_[i]<events_[j]? i: j;
        return;
    }

    // handle the general case of more than 4 lanes
    int p = (nodes_+lane-1)/2;
    unsigned olane = lane^1;
    index_tree_[p] = events_[lane]<events_[olane] ? lane: olane;
    while (p) {
        const auto l1 = index_tree_[p];
        const auto l2 = index_tree_[p-1+2*(p&1)];
        p = (p-1)>>1; // p = parent(p)
        index_tree_[p] = events_[l1]<events_[l2] ? l1: l2;
    }
}

void tourney_tree::setup(unsigned i) {
    if (is_leaf(i)) {
        const auto l = left(i)-nodes_;
        const auto r = l+1;
        // set index_tree_[i] to the lane index of the highest priority child lane
        index_tree_[i] = events_[l]<events_[r] ? l: r;
        return;
    }
    const auto l = left(i);
    const auto r = l+1;
    setup(l);
    setup(r);
    const auto li = index_tree_[l];
    const auto ri = index_tree_[r];
    index_tree_[i] = events_[li]<events_[ri]? li: ri;
};

// The tree is stored using the standard heap indexing scheme.

inline bool tourney_tree::is_leaf(unsigned i) const {
    return i >= nodes_/2;
}
// tree position of left child of node i
inline unsigned tourney_tree::left(unsigned i) const {
    return (i<<1) + 1;
}
postsynaptic_spike_event& tourney_tree::event(unsigned i) {
    return events_[index_tree_[i]];
}
inline
const postsynaptic_spike_event& tourney_tree::event(unsigned i) const {
    return events_[index_tree_[i]];
}

inline
unsigned tourney_tree::next_power_2(unsigned x) const {
    unsigned n = 1;
    while (n<x) n<<=1;
    return n;
}

} // namespace impl

void merge_events(time_type t0, time_type t1,
                  const pse_vector& lc, pse_vector& events,
                  std::vector<event_generator>& generators,
                  pse_vector& lf)
{
    using std::distance;
    using std::lower_bound;

    // Sort events from the communicator in place.
    util::sort(events);

    // Clear lf to store merged list.
    lf.clear();

    // Merge the incoming event sequences into a single vector in lf
    if (generators.size()) {
        // Handle the case where the cell has event generators.
        // This is performed in two steps:
        //  Step 1 : Use tournament tree to merge all events in lc, events and
        //           generators to be delivered in the time interval [t₀, t₁).
        //  Step 2 : Use std::merge to append events in lc and events with
        //           delivery times in the interval [t₁, ∞).
        EXPECTS(generators.size()>2u);

        // Make an event generator with all the events in events.
        generators[0] = seq_generator<pse_vector>(events);

        // Make an event generator with all the events in lc with time >= t0
        auto lc_it = lower_bound(lc.begin(), lc.end(), t0, event_time_less());
        auto lc_range = util::make_range(lc_it, lc.end());
        generators[1] = seq_generator<decltype(lc_range)>(lc_range);

        // Perform k-way merge of all events in events, lc and the generators
        // that are due to be delivered in the interval [t₀, t₁)
        impl::tourney_tree tree(generators);
        while (!tree.empty(t1)) {
            lf.push_back(tree.head());
            tree.pop();
        }

        // Find first event in lc with delivery time >= t1
        lc_it = lower_bound(lc.begin(), lc.end(), t1, event_time_less());
        // Find first event in events with delivery time >= t1
        auto ev_it = lower_bound(events.begin(), events.end(), t1, event_time_less());
        const auto m = lf.size();
        const auto n = m + distance(lc_it, lc.end()) + distance(ev_it, events.end());
        lf.resize(n);
        std::merge(ev_it, events.end(), lc_it, lc.end(), lf.begin()+m);

        // clear the generators associated with temporary event sequences
        generators[0] = generators[1] = event_generator();
    }
    else {
        // Handle the case where the cell has no event generators: only events
        // in lc and lf with delivery times >= t0 must be merged, which can be
        // handles with a single call to std::merge.
        auto pos = std::lower_bound(lc.begin(), lc.end(), t0, event_time_less());
        lf.resize(events.size()+distance(pos, lc.end()));
        std::merge(events.begin(), events.end(), pos, lc.end(), lf.begin());
    }
}

} // namespace arb

