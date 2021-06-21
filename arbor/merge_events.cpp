#include <iostream>
#include <set>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/math.hpp>

#include "io/trace.hpp"
#include "merge_events.hpp"
#include "profile/profiler_macro.hpp"

namespace arb {

namespace impl {

// A postsynaptic spike event that has delivery time set to
// terminal_time, used as a sentinel in `tourney_tree`.

static constexpr spike_event terminal_pse{0, terminal_time, 0};


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

tourney_tree::tourney_tree(std::vector<event_span>& input):
    input_(input),
    n_lanes_(input_.size())
{
    // Must have at least 1 queue.
    arb_assert(n_lanes_>=1u);

    leaves_ = math::next_pow2(n_lanes_);

    // Must be able to fit leaves in unsigned count.
    arb_assert(leaves_>=n_lanes_);
    nodes_ = 2*leaves_-1;

    // Allocate space for the tree nodes
    heap_.resize(nodes_);
    // Set the leaf nodes
    for (auto i=0u; i<leaves_; ++i) {
        heap_[leaf(i)] = i<n_lanes_?
            key_val(i, input[i].empty()? terminal_pse: input[i].front()):
            key_val(i, terminal_pse); // null leaf node
    }
    // Walk the tree to initialize the non-leaf nodes
    setup(0);
}

std::ostream& operator<<(std::ostream& out, const tourney_tree& tt) {
    unsigned nxt = 1;
    for (unsigned i = 0; i<tt.nodes_; ++i) {
        if (i==nxt-1) {
            nxt*=2;
            out << "\n";
        }
        out << "{" << tt.heap_[i].first << "," << tt.heap_[i].second << "}\n";
    }
    return out;
}

bool tourney_tree::empty() const {
    return event(0).time == terminal_time;
}

spike_event tourney_tree::head() const {
    return event(0);
}

// Remove the smallest (most recent) event from the tree, then update the
// tree so that head() returns the next event.
void tourney_tree::pop() {
    unsigned lane = id(0);
    unsigned i = leaf(lane);

    // draw the next event from the input lane
    auto& in = input_[lane];

    if (!in.empty()) {
        ++in.left;
    }

    event(i) = in.empty()? terminal_pse: in.front();

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
spike_event& tourney_tree::event(unsigned i) {
    return heap_[i].second;
}
const spike_event& tourney_tree::event(unsigned i) const {
    return heap_[i].second;
}

} // namespace impl

void tree_merge_events(std::vector<event_span>& sources, pse_vector& out) {
    impl::tourney_tree tree(sources);
    while (!tree.empty()) {
        out.push_back(tree.head());
        tree.pop();
    }
}

} // namespace arb

