#include "tourney_tree.hpp"

#include <iostream>
#include <numeric>

#include <arbor/math.hpp>
#include <arbor/assert.hpp>

namespace arb {

using event_span = util::range<const spike_event*>;

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

tourney_tree::tourney_tree(std::vector<event_span>&& input):
    input_(std::move(input)),
    n_lanes_(input_.size()),
    leaves_{math::next_pow2(n_lanes_)},
    nodes_{2*leaves_ - 1} {
    // Must have at least 1 queue.
    arb_assert(!input_.empty());
    // Must be able to fit leaves in unsigned count.
    arb_assert(leaves_ >= n_lanes_);
    // Allocate space for the tree nodes
    heap_vals_.resize(nodes_, terminal_pse);
    heap_keys_.resize(nodes_);
    // Set the leaf nodes
    for (auto i=0u; i<n_lanes_; ++i) {
        auto idx = leaf(i);
        if (!input_[i].empty()) heap_vals_[idx] = input_[i].front();
        heap_keys_[idx] = i;
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
        out << "{" << tt.heap_keys_[i] << "," << tt.heap_vals_[i] << "}\n";
    }
    return out;
}

bool tourney_tree::empty() const { return head().time == terminal_time; }
spike_event tourney_tree::head() const { return event(0); }

// Remove the smallest (most recent) event from the tree, then update the
// tree so that head() returns the next event.
spike_event tourney_tree::pop() {
    spike_event evt = heap_vals_[0];
    auto lane = heap_keys_[0];
    auto idx = leaf(lane);

    // draw the next event from the input lane
    auto& in = input_[lane];

    if (!in.empty()) ++in.left;

    heap_vals_[idx] = in.empty() ? terminal_pse : in.front();

    // re-heapify the tree with a single walk from leaf to root
    while ((idx = parent(idx))) merge_up(idx);
    // handle the root
    merge_up(0);
    return evt;
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
    if (event(l)<event(r)) {
        heap_keys_[i] = heap_keys_[l];
        heap_vals_[i] = heap_vals_[l];
    }
    else {
        heap_keys_[i] = heap_keys_[r];
        heap_vals_[i] = heap_vals_[r];
    }
}

std::size_t tourney_tree::size() const { return leaves_; }

// The tree is stored using the standard heap indexing scheme.

unsigned tourney_tree::parent(unsigned i) const { return (i-1)>>1; }
unsigned tourney_tree::left(unsigned i) const { return (i<<1) + 1; }
unsigned tourney_tree::right(unsigned i) const { return left(i)+1; }
unsigned tourney_tree::leaf(unsigned i) const { return i+leaves_-1;}
bool tourney_tree::is_leaf(unsigned i) const { return i>=leaves_-1; }
const unsigned& tourney_tree::id(unsigned i) const { return heap_keys_[i] ;}
spike_event& tourney_tree::event(unsigned i) { return heap_vals_[i]; }
const spike_event& tourney_tree::event(unsigned i) const { return heap_vals_[i]; }

void tree_merge_events(std::vector<event_span> sources, pse_vector& out) {
    tourney_tree tree(std::move(sources));
    while (!tree.empty()) out.emplace_back(tree.pop());
}

} // arb
