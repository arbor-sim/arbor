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
tourney_tree::tourney_tree(std::vector<event_generator_ptr>& input):
    input_(input),
    n_lanes_(input_.size())
{
    // Must have at least 1 queue
    EXPECTS(n_lanes_);
    // Maximum value in unsigned limits how many queues we can have
    EXPECTS(n_lanes_<(1u<<(sizeof(unsigned)*8u-1u)));

    leaves_ = next_power_2(n_lanes_);
    nodes_ = 2u*(leaves_-1u)+1u; // 2*l-1 with overflow protection

    // Allocate space for the tree nodes
    heap_.resize(nodes_);
    // Set the leaf nodes
    for (auto i=0u; i<leaves_; ++i) {
        heap_[leaf(i)] = i<n_lanes_?
            key_val(i, input[i]->next()):
            key_val(i, terminal_pse()); // null leaf node
    }
    // Walk the tree to initialize the non-leaf nodes
    setup(0);
}

void tourney_tree::print() const {
    auto nxt=1u;
    for (auto i=0u; i<nodes_; ++i) {
        if (i==nxt-1) { nxt*=2; std::cout << "\n";}
        std::cout << "{" << heap_[i].first << "," << heap_[i].second << "}\n";
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
    unsigned lane = id(0);
    unsigned i = leaf(lane);
    // draw the next event from the input lane
    input_[lane]->pop();
    // place event the leaf node for this lane
    event(i) = input_[lane]->next();

    // re-heapify the tree with a single walk from leaf to root
    while ((i=parent(i))) {
        merge_up(i);
    }
    merge_up(0); // handle the root
}

void tourney_tree::setup(unsigned i) {
    if (is_leaf(i)) return;
    setup(left(i));
    setup(right(i));
    merge_up(i);
};

// Update the value at node i of the tree to be the smallest
// of its left and right children.
// The result is undefined for leaf nodes.
void tourney_tree::merge_up(unsigned i) {
    const auto l = left(i);
    const auto r = right(i);
    heap_[i] = event(l)<event(r)? heap_[l]: heap_[r];
}

// The tree is stored using the standard heap indexing scheme.

unsigned tourney_tree::parent(unsigned i) const {
    return (i-1)>>1;
}
unsigned tourney_tree::left(unsigned i) const {
    return (i<<1) + 1;
}
unsigned tourney_tree::right(unsigned i) const {
    return left(i)+1;
}
unsigned tourney_tree::leaf(unsigned i) const {
    return i+leaves_-1;
}
bool tourney_tree::is_leaf(unsigned i) const {
    return i>=leaves_-1;
}
const unsigned& tourney_tree::id(unsigned i) const {
    return heap_[i].first;
}
postsynaptic_spike_event& tourney_tree::event(unsigned i) {
    return heap_[i].second;
}
const postsynaptic_spike_event& tourney_tree::event(unsigned i) const {
    return heap_[i].second;
}

unsigned tourney_tree::next_power_2(unsigned x) const {
    unsigned n = 1;
    while (n<x) n<<=1;
    return n;
}

} // namespace impl

void merge_events(time_type t0, time_type t1,
                  const pse_vector& lc, pse_vector& events,
                  std::vector<event_generator_ptr>& generators,
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
        generators[0] = make_event_generator<seq_generator<pse_vector>>(events);

        // Make an event generator with all the events in lc with time >= t0
        auto lc_it = lower_bound(lc.begin(), lc.end(), t0, event_time_less());
        auto lc_range = util::make_range(lc_it, lc.end());
        generators[1] = make_event_generator<seq_generator<decltype(lc_range)>>(lc_range);

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

