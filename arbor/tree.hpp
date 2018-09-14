#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>
#include <vector>

#include <arbor/common_types.hpp>

#include "algorithms.hpp"
#include "memory/memory.hpp"
#include "util/span.hpp"

namespace arb {

class tree {
public:
    using int_type   = cell_lid_type;
    using size_type  = cell_local_size_type;

    using iarray = std::vector<int_type>;
    static constexpr int_type no_parent = (int_type)-1;

    tree() = default;

    /// Create the tree from a parent index that lists the parent segment
    /// of each segment in a cell tree.
    tree(std::vector<int_type> parent_index);

    size_type num_children() const;

    size_type num_children(size_t b) const;

    size_type num_segments() const;

    /// return the child index
    const iarray& child_index();

    /// return the list of all children
    const iarray& children() const;

    /// return the list of all children of branch i
    auto children(size_type i) const {
        const auto b = child_index_[i];
        const auto e = child_index_[i+1];
        return util::subrange_view(children_, b, e);
    }

    /// return the list of parents
    const iarray& parents() const;

    /// return the parent of branch b
    const int_type& parent(size_t b) const;
    int_type& parent(size_t b);

    // splits the node in two parts. Returns `ix + 1` which is the new index of
    // the old node.
    // .-------------------.
    // |      P         P  |
    // |     /         /   |
    // |    A   ~~>   N    |
    // |   / \        |    |
    // |  B  C        A    |
    // |             / \   |
    // |            B  C   |
    // '-------------------'
    int_type split_node(int_type ix);

    // Changes the root node of a tree
    // .------------------------.
    // |        A               |
    // |       / \         R    |
    // |      R  B        / \   |
    // |     /     ~~>   A  C   |
    // |    C           /  / \  |
    // |   / \         B  D  E  |
    // |  D  E                  |
    // '------------------------'
    // Returns the permutation applied to the nodes,
    // i.e. `new_node[i] = old_node[perm[i]]`
    iarray select_new_root(int_type root);

    /// memory used to store tree (in bytes)
    std::size_t memory() const;


    // Exports the tree's parent structure into a dot file.
    // the children and parent trees are equivalent. Both methods are provided
    // for debugging purposes.
    void export_parents(std::string file) const {
        // the labels in the the graph are the branch indices
        export_parents(file, [](auto i){return i;});
    }

    template<typename F>
    void export_parents(std::string file, F label) const {
        using util::make_span;
        std::ofstream ofile;
        ofile.open(file);
        ofile << "strict digraph Parents {" << std::endl;
        for (auto i: make_span(parents().size())) {
            ofile << i << "[label=\"" << label(i) << "\"]" << std::endl;
        }
        for (auto i: make_span(parents().size())) {
            auto p = parent(i);
            if (p != no_parent) {
                ofile << i << " -> " << parent(i) << std::endl;
            }
        }
        ofile << "}" << std::endl;
        ofile.close();
    }

    // Exports the tree's children structure into a dot file.
    // the children and parent trees are equivalent. Both methods are provided
    // for debugging purposes.
    void export_children(std::string file) const {
        // the labels in the the graph are the branch indices
        export_children(file, [](auto i){return i;});
    }

    template<typename F>
    void export_children(std::string file, F label) const {
        using util::make_span;
        std::ofstream ofile;
        ofile.open(file);
        ofile << "strict digraph Children {" << std::endl;
        for (auto i: make_span(child_index_.size() - 1)) {
            ofile << i << "[label=\"" << label(i) << "\"]" << std::endl;
        }
        for (auto i: make_span(child_index_.size() - 1)) {
            ofile << i << " -> {";
            for (auto c: children(i)) {
                ofile << " " << c;
            }
            ofile << "}" << std::endl;
        }
        ofile << "}" << std::endl;
        ofile.close();
    }

private:
    void init(size_type nnode);

    // state
    iarray children_;
    iarray child_index_;
    iarray parents_;
};

// Calculates the depth of each branch from the root of a cell segment tree.
// The root has depth 0, it's children have depth 1, and so on.
tree::iarray depth_from_root(const tree& t);

template <typename C>
std::vector<tree::int_type> make_parent_index(tree const& t, C const& counts)
{
    using util::make_span;
    using int_type = tree::int_type;
    constexpr auto no_parent = tree::no_parent;

    if (!algorithms::all_positive(counts) || counts.size() != t.num_segments()) {
        throw std::domain_error(
            "make_parent_index requires one non-zero count per segment"
        );
    }
    auto index = algorithms::make_index(counts);
    auto num_compartments = index.back();
    std::vector<int_type> parent_index(num_compartments);
    int_type pos = 0;
    for (int_type i : make_span(0, t.num_segments())) {
        // get the parent of this segment
        // taking care for the case where the root node has -1 as its parent
        auto parent = t.parent(i);
        parent = parent!=no_parent ? parent : 0;

        // the index of the first compartment in the segment
        // is calculated differently for the root (i.e when i==parent)
        if (i!=parent) {
            parent_index[pos++] = index[parent+1]-1;
        }
        else {
            parent_index[pos++] = parent;
        }
        // number the remaining compartments in the segment consecutively
        while (pos<index[i+1]) {
            parent_index[pos] = pos-1;
            pos++;
        }
    }

    // if one of these assertions is tripped, we have to improve
    // the input validation above
    assert(pos==num_compartments);
    assert(algorithms::is_minimal_degree(parent_index));

    return parent_index;
}

} // namespace arb
