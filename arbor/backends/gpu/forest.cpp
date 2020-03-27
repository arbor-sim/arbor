#include "backends/gpu/forest.hpp"
#include "tree.hpp"
#include "util/span.hpp"

namespace arb {
namespace gpu {

forest::forest(const std::vector<size_type>& p, const std::vector<size_type>& cell_cv_divs) :
    perm_balancing(p.size())
{
    using util::make_span;

    auto num_cells = cell_cv_divs.size() - 1;

    for (auto c: make_span(0u, num_cells)) {
        // build the parent index for cell c
        auto cell_start = cell_cv_divs[c];
        std::vector<unsigned> cell_p =
            util::assign_from(
                util::transform_view(
                    util::subrange_view(p, cell_cv_divs[c], cell_cv_divs[c+1]),
                    [cell_start](unsigned i) {return i == -1 ? i : i - cell_start;}));

        auto fine_tree = tree(cell_p);

        // select a root node and merge branches with discontinuous compartment
        // indices
        auto perm = fine_tree.select_new_root(0);
        for (auto i: make_span(perm.size())) {
            perm_balancing[cell_start + i] = cell_start + perm[i];
        }

        // find the index of the first node for each branch
        auto branch_starts = algorithms::branches(fine_tree.parents());

        // compute branch length and apply permutation
        std::vector<unsigned> branch_lengths(branch_starts.size() - 1);
        for (auto i: make_span(branch_lengths.size())) {
            branch_lengths[i] = branch_starts[i+1] - branch_starts[i];
        }

        // find the parent index of branches
        // we need to convert to cell_lid_type, required to construct a tree.
        std::vector<cell_lid_type> branch_p =
            util::assign_from(
                algorithms::tree_reduce(fine_tree.parents(), branch_starts));
        // build tree structure that describes the branch topology
        auto cell_tree = tree(branch_p);

        trees.push_back(cell_tree);
        fine_trees.push_back(fine_tree);
        tree_branch_starts.push_back(branch_starts);
        tree_branch_lengths.push_back(branch_lengths);
    }
}

void forest::optimize() {
    using util::make_span;

    // cut the tree
    unsigned count = 1; // number of nodes found on the previous level
    for (auto level = 0; count > 0; level++) {
        count = 0;

        // decide where to cut it ...
        unsigned max_length = 0;
        for (auto t_ix: make_span(trees.size())) { // TODO make this local on an intermediate packing
            for (level_iterator it (&trees[t_ix], level); it.valid(); it.next()) {
                auto length = tree_branch_lengths[t_ix][it.peek()];
                max_length += length;
                count++;
            }
        }
        if (count == 0) {
            // there exists no tree with branches on this level
            continue;
        };
        max_length = max_length / count;
        // avoid ininite loops
        if (max_length <= 1) max_length = 1;
        // we don't want too small segments
        if (max_length <= 10) max_length = 10;

        for (auto t_ix: make_span(trees.size())) {
            // ... cut all trees on this level
            for (level_iterator it (&trees[t_ix], level); it.valid(); it.next()) {

                auto length = tree_branch_lengths[t_ix][it.peek()];
                if (length > max_length) {
                    // now cut the tree

                    // we are allowed to mess with the tree because of the
                    // implementation of level_iterator o.O

                    auto insert_at_bs = tree_branch_starts[t_ix].begin() + it.peek();
                    auto insert_at_ls = tree_branch_lengths[t_ix].begin() + it.peek();

                    trees[t_ix].split_node(it.peek());

                    // now the tree got a new node.
                    // we now have to insert a corresponding new 'branch
                    // start' to the list

                    // make sure that `tree_branch_starts` for A and N point to
                    // the correct slices
                    auto old_start = tree_branch_starts[t_ix][it.peek()];
                    // first insert, then index peek, as we already
                    // incremented the iterator
                    tree_branch_starts[t_ix].insert(insert_at_bs, old_start);
                    tree_branch_lengths[t_ix].insert(insert_at_ls, max_length);
                    tree_branch_starts[t_ix][it.peek() + 1] = old_start + max_length;
                    tree_branch_lengths[t_ix][it.peek() + 1] = length - max_length;
                    // we don't have to shift any indices as we did not
                    // create any new branch segments, but just split
                    // one over two nodes
                }
            }
        }
    }
}


// debugging functions:


// Exports the tree's parent structure into a dot file.
// the children and parent trees are equivalent. Both methods are provided
// for debugging purposes.
template<typename F>
void export_parents(const tree& t, std::string file, F label) {
    using util::make_span;
    std::ofstream ofile;
    ofile.open(file);
    ofile << "strict digraph Parents {" << std::endl;
    for (auto i: make_span(t.parents().size())) {
        ofile << i << "[label=\"" << label(i) << "\"]" << std::endl;
    }
    for (auto i: make_span(t.parents().size())) {
        auto p = t.parent(i);
        if (p != tree::no_parent) {
            ofile << i << " -> " << t.parent(i) << std::endl;
        }
    }
    ofile << "}" << std::endl;
    ofile.close();
}

void export_parents(const tree& t, std::string file) {
    // the labels in the the graph are the branch indices
    export_parents(t, file, [](auto i){return i;});
}

// Exports the tree's children structure into a dot file.
// the children and parent trees are equivalent. Both methods are provided
// for debugging purposes.
template<typename F>
void export_children(const tree& t, std::string file, F label) {
    using util::make_span;
    std::ofstream ofile;
    ofile.open(file);
    ofile << "strict digraph Children {" << std::endl;
    for (auto i: make_span(t.num_segments())) {
        ofile << i << "[label=\"" << label(i) << "\"]" << std::endl;
    }
    for (auto i: make_span(t.num_segments())) {
        ofile << i << " -> {";
        for (auto c: t.children(i)) {
            ofile << " " << c;
        }
        ofile << "}" << std::endl;
    }
    ofile << "}" << std::endl;
    ofile.close();
}

void export_children(const tree& t, std::string file) {
    // the labels in the the graph are the branch indices
    export_children(t, file, [](auto i){return i;});
}

} // namespace gpu
} // namespace arb
