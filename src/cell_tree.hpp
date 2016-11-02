#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

#include <memory/memory.hpp>

#include "common_types.hpp"
#include "tree.hpp"

namespace nest {
namespace mc {

/// The tree data structure that describes the segments of a cell tree.
/// A cell is represented as a tree where each node may have any number of
/// children. Typically in a cell only the soma has more than two segments,
/// however this rule does not appear to be strictly followed in the PCP data
/// sets.
///
/// To optimize some computations it is important the the tree be balanced,
/// in the sense that the depth of the tree be minimized. This means that it is
/// necessary that any node in the tree may be used as the root. In the PCP data
/// sets it appears that the soma was always index 0, however we need more
/// flexibility in choosing the root.
class cell_tree {
    using range = memory::Range;

public:
    using int_type        = cell_lid_type;
    using size_type       = cell_local_size_type;

    using iarray      = memory::host_vector<int_type>;
    using view_type       = iarray::view_type;
    using const_view_type = iarray::const_view_type;

    using tree = nest::mc::tree<int_type, size_type>;
    static constexpr int_type no_parent = tree::no_parent;

    /// default empty constructor
    cell_tree() = default;

    /// construct from a parent index
    cell_tree(std::vector<int_type> const& parent_index)
    {
        // handle the case of an empty parent list, which implies a single-compartment model
        if(parent_index.size()>0) {
            tree_ = tree(parent_index);
        }
        else {
            tree_ = tree(std::vector<int_type>({0}));
        }
    }

    /// construct from a tree
    // copy constructor
    cell_tree(tree const& t, int_type s)
    : tree_(t),
      soma_(s)
    { }

    // move constructor
    cell_tree(tree&& t, int_type s)
    : tree_(std::move(t)),
      soma_(s)
    { }

    /// construct from a cell tree
    // copy constructor
    cell_tree(cell_tree const& other)
    : tree_(other.tree_),
      soma_(other.soma())
    { }

    // assignment from rvalue
    cell_tree& operator=(cell_tree&& other)
    {
        std::swap(other.tree_, tree_);
        std::swap(other.soma_, soma_);
        return *this;
    }

    // assignment
    cell_tree& operator=(cell_tree const& other)
    {
        tree_ = other.tree_;
        soma_ = other.soma_;
        return *this;
    }

    // move constructor
    cell_tree(cell_tree&& other)
    {
        *this = std::move(other);
    }

    tree const& graph() const {
        return tree_;
    }

    int_type soma() const {
        return soma_;
    }

    /// Minimize the depth of the tree.
    int_type balance() {
        // find the new root
        auto new_root = find_minimum_root();

        // change the root on the tree
        auto p = tree_.change_root(new_root);

        // keep track of the soma_
        if(p.size()) {
            soma_ = p[soma_];
        }

        return new_root;
    }

    /// memory used to store cell tree (in bytes)
    size_t memory() const {
        return tree_.memory() + sizeof(cell_tree) - sizeof(tree);
    }

    /// returns the number of segments in the cell
    size_t num_segments() const {
        return tree_.num_nodes();
    }

    /// returns the number of child segments of segment b
    size_type num_children(int_type b) const {
        return tree_.num_children(b);
    }

    /// returns a list of the children of segment b
    const_view_type children(int_type b) const {
        return tree_.children(b);
    }

    /// returns the parent index of segment b
    int_type parent(size_t b) const {
        return tree_.parent(b);
    }

    /// writes a graphviz .dot file that visualizes cell segment structure
    void to_graphviz(std::string const& fname) const {

        std::ofstream fid(fname);

        fid << "graph cell {" << '\n';
        for(auto b : range(0,num_segments())) {
            if(children(b).size()) {
                for(auto c : children(b)) {
                    fid << "  " << b << " -- " << c << ";" << '\n';
                }
            }
        }
        fid << "}" << std::endl; // flush at end of output?
    }

    iarray depth_from_leaf()
    {
        tree::iarray depth(num_segments());
        depth_from_leaf(depth, int_type{0});
        return depth;
    }

    iarray depth_from_root()
    {
        tree::iarray depth(num_segments());
        depth[0] = 0;
        depth_from_root(depth, int_type{1});
        return depth;
    }

private :

    /// helper type for sub-tree computation
    /// use in balance()
    struct sub_tree {
        sub_tree(int_type r, int_type diam, int_type dpth):
            root(r), diameter(diam), depth(dpth)
        {}

        void set(int r, int diam, int dpth) {
            root = r;
            diameter = diam;
            depth = dpth;
        }

        std::string to_string() const {
            return
               "[" + std::to_string(root) + ","
                   + std::to_string(diameter)  + ","
                   + std::to_string(depth) +
               "]";
        }

        int_type root;
        int_type diameter;
        int_type depth;
    };

    /// returns the index of the segment that would minimise the depth of the
    /// tree if used as the root segment
    int_type find_minimum_root() {
        if (num_segments()==1) {
            return 0;
        }

        // calculate the depth of each segment from the root
        //      pre-order traversal of the tree
        auto depth = depth_from_root();

        auto max       = std::max_element(depth.begin(), depth.end());
        auto max_leaf  = std::distance(depth.begin(), max);

        // Calculate the depth of each compartment as the maximum distance
        // from a child leaf
        //      post-order traversal of the tree
        depth = depth_from_leaf();

        // Walk back from the deepest leaf towards the root.
        // At each node test to find the deepest sub-tree that doesn't include
        // node max_leaf. Then check if the total diameter of this sub-tree and
        // the sub-tree and node max_leaf is the largest found so far.  When the
        // walk has been completed to the root node, the node that has been
        // selected will be the root of the sub-tree with the largest diameter.
        sub_tree max_sub_tree(0, 0, 0);
        int_type distance_from_max_leaf = 1;
        auto pnt = max_leaf;
        auto pos = parent(max_leaf);
        while(pos != no_parent) {
            for(auto c : children(pos)) {
                if(c!=pnt) {
                    auto diameter = depth[c] + 1 + distance_from_max_leaf;
                    if (diameter>max_sub_tree.diameter) {
                        max_sub_tree.set(pos, diameter, distance_from_max_leaf);
                    }
                }
            }
            pnt = pos;
            pos = parent(pos);
            ++distance_from_max_leaf;
        }

        // calculate the depth of the balanced tree
        auto new_depth = (max_sub_tree.diameter+1) / 2;

        // nothing to do if the current root is also the root of the
        // balanced tree
        if(max_sub_tree.root==0 && max_sub_tree.depth==new_depth) {
            return 0;
        }

        // perform another walk from max leaf towards max_sub_tree.root
        auto count = new_depth;
        auto new_root = max_leaf;
        while(count) {
            new_root = parent(new_root);
            --count;
        }

        return new_root;
    }

    int_type depth_from_leaf(iarray& depth, int_type segment)
    {
        int_type max_depth = 0;
        for(auto c : children(segment)) {
            max_depth = std::max(max_depth, depth_from_leaf(depth, c));
        }
        depth[segment] = max_depth;
        return max_depth+1;
    }

    void depth_from_root(iarray& depth, int_type segment)
    {
        auto d = depth[parent(segment)] + 1;
        depth[segment] = d;
        for(auto c : children(segment)) {
            depth_from_root(depth, c);
        }
    }

    //////////////////////////////////////////////////
    // state
    //////////////////////////////////////////////////

    /// storage for the tree structure of cell segments
    tree tree_;

    /// index of the soma
    int_type soma_ = 0;
};

} // namespace mc
} // namespace nest
