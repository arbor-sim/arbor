#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>
#include <memory>

#include "vector/include/Vector.hpp"
#include "tree.hpp"
#include "util.hpp"

// The tree data structure that describes the branches of a cell tree.
// A cell is represented as a tree where each node may have any number of
// children. Typically in a cell only the soma has more than two branches,
// however this rule does not appear to be strictly followed in the PCP data
// sets.
//
// To optimize some computations it is important the the tree be balanced,
// in the sense that the depth of the tree be minimized. This means that it is
// necessary that any node in the tree may be used as the root. In the PCP data
// sets it appears that the soma was always index 0, however we need more
// flexibility in choosing the root.
class cell_tree {
    public :

    // use a signed 16-bit integer for storage of indexes, which is reasonable given
    // that typical cells have at most 1000-2000 nodes
    using int_type = int16_t;
    using index_type = memory::HostVector<int_type>;
    using index_view = index_type::view_type;

    /// construct from a parent index
    cell_tree(std::vector<int> const& parent_index)
    {
        // handle the case of an empty parent list, which implies a single-compartment model
        std::vector<int> branch_index;
        if(parent_index.size()>0) {
            branch_index = tree_.init_from_parent_index(parent_index);
        }
        else {
            branch_index = tree_.init_from_parent_index(std::vector<int>({0}));
        }

        // if needed, calculate meta-data like length[] and end[] arrays for data
    }

    /// Minimize the depth of the tree.
    /// Pick a root node that minimizes the depth of the tree.
    /*
    int_type balance() {
        if(num_branches()==1) {
            return 0;
        }

        // calculate the depth of each branch from the root
        //      pre-order traversal of the tree
        index_type depth(num_branches());
        auto depth_from_root = [this, &depth] (int_type b) -> void
        {
            auto d = depth[tree.parent(b)] + 1;
            depth[b] = d;
            for(auto c : tree.children(b)) {
                depth_from_root(c);
            }
        }
        depth[0]=0;
        depth_from_root(0);

        auto max       = std::max_element(depth.begin(), depth.end());
        auto max_leaf  = std::distance(depth.begin(), max);
        auto original_depth = *max;

        // Calculate the depth of each compartment as the maximum distance
        // from a child leaf
        //      post-order traversal of the tree
        auto depth_from_leaf = [this, &depth] (int_type b)
        {
            int_type max_depth = 0;
            for(auto c : children(branch)) {
                max_depth = std::max(max_depth, depth_from_leaf(c));
            }
            depth[b] = max_depth;
            return max_depth+1;
        }
        depth_from_leaf(0);

        // Walk back from the deepest leaf towards the root.
        // At each node test to find the deepest sub-tree that doesn't include
        // node max_leaf. Then check if the total diameter of this sub-tree and
        // the sub-tree and node max_leaf is the largest found so far.  When the
        // walk has been completed to the root node, the node that has been
        // selected will be the root of the sub-tree with the largest diameter.
        sub_tree max_sub_tree(0, 0, 0);
        auto distance_from_max_leaf = 1;
        auto parent = max_leaf;
        auto pos = parent(max_leaf);
        while(pos != -1) {
            for(auto c : children(pos)) {
                if(c!=parent) {
                    auto diameter = depth[c] + 1 + distance_from_max_leaf;
                    if(diameter>max_sub_tree.diameter) {
                        max_sub_tree.set(pos, diameter, distance_from_max_leaf);
                    }
                }
            }
            parent = pos;
            pos = parent(pos);
            ++distance_from_max_leaf;
        }

        std::cout << memory::util::green(std::to_string(max_sub_tree.root)) << " ";

        // calculate the depth of the balanced tree
        auto new_depth = (max_sub_tree.diameter+1) / 2;

        // nothing to do if the current root is also the root of the
        // balanced tree
        if(max_sub_tree.root==0 && max_sub_tree.depth==new_depth) {
            std::cout << " root " << 0 << " depth " << original_depth << std::endl;
            return *max;
        }

        // perform another walk from max leaf towards max_sub_tree.root
        auto count = new_depth;
        auto new_root = max_leaf;
        while(count) {
            new_root = parent(new_root);
            --count;
        }

        // change the root on the tree
        tree.change_root(new_root);

        return new_depth;
    }
    */

    /// memory used to store cell tree (in bytes)
    size_t memory() const {
        return tree_.memory() + sizeof(cell_tree) - sizeof(tree);
    }

    /// returns the number of branches in the cell
    size_t num_branches() const {
        return tree_.num_nodes();
    }

    /// returns the number of child branches of branch b
    size_t num_children(size_t b) const {
        return tree_.num_children(b);
    }

    /// returns a list of the children of branch b
    const index_view children(size_t b) const {
        return tree_.children(b);
    }

    /// returns the parent of branch b
    int_type parent(size_t b) const {
        return tree_.parent(b);
    }

    /// generates a graphviz .dot file that visualizes cell branch structure
    void to_graphviz(std::string const& fname) const {

        std::ofstream fid(fname);

        fid << "graph cell {" << std::endl;
        for(auto b : range(0,num_branches())) {
            if(children(b).size()) {
                for(auto c : children(b)) {
                    fid << "  " << b << " -- " << c << ";" << std::endl;
                }
            }
        }
        fid << "}" << std::endl;
    }

    private :


    /// helper type for sub-tree computation
    /// use in balance()
    struct sub_tree {
        sub_tree(int r, int diam, int dpth)
        : root(r), diameter(diam), depth(dpth)
        {}

        void set(int r, int diam, int dpth)
        {
            root = r;
            diameter = diam;
            depth = dpth;
        }

        std::string to_string() const {
            std::string s;

            s += "[" + std::to_string(root) + ","
                + std::to_string(diameter)  + "," + std::to_string(depth) + "]";

            return s;
        }

        int root;
        int diameter;
        int depth;
    };

    // storage for the tree structure of cell branches
    tree tree_;
};

