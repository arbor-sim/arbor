#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

#include "vector/include/Vector.hpp"
#include "tree.hpp"
#include "util.hpp"

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
    public :

    // use a signed 16-bit integer for storage of indexes, which is reasonable given
    // that typical cells have at most 1000-2000 segments
    using int_type = int16_t;
    using index_type = memory::HostVector<int_type>;
    using index_view = index_type::view_type;

    /// default empty constructor
    cell_tree() = default;

    /// construct from a parent index
    cell_tree(std::vector<int> const& parent_index)
    {
        // handle the case of an empty parent list, which implies a single-compartment model
        std::vector<int> segment_index;
        if(parent_index.size()>0) {
            segment_index = tree_.init_from_parent_index(parent_index);
        }
        else {
            segment_index = tree_.init_from_parent_index(std::vector<int>({0}));
        }

        // if needed, calculate meta-data like length[] and end[] arrays for data
    }

    /// construct from a tree
    // copy constructor
    cell_tree(tree const& t, int s)
    : tree_(t),
      soma_(s)
    { }

    // move constructor
    cell_tree(tree&& t, int s)
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
    size_t num_children(size_t b) const {
        return tree_.num_children(b);
    }

    /// returns a list of the children of segment b
    const index_view children(size_t b) const {
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

    index_type depth_from_leaf()
    {
        tree::index_type depth(num_segments());
        depth_from_leaf(depth, 0);
        return depth;
    }

    index_type depth_from_root()
    {
        tree::index_type depth(num_segments());
        depth[0] = 0;
        depth_from_root(depth, 1);
        return depth;
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
            return
               "[" + std::to_string(root) + ","
                   + std::to_string(diameter)  + ","
                   + std::to_string(depth) +
               "]";
        }

        int root;
        int diameter;
        int depth;
    };

    /// returns the index of the segment that would minimise the depth of the
    /// tree if used as the root segment
    int_type find_minimum_root() {
        if(num_segments()==1) {
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
        auto distance_from_max_leaf = 1;
        auto pnt = max_leaf;
        auto pos = parent(max_leaf);
        while(pos != -1) {
            for(auto c : children(pos)) {
                if(c!=pnt) {
                    auto diameter = depth[c] + 1 + distance_from_max_leaf;
                    if(diameter>max_sub_tree.diameter) {
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

    int_type depth_from_leaf(index_type& depth, int segment)
    {
        int_type max_depth = 0;
        for(auto c : children(segment)) {
            max_depth = std::max(max_depth, depth_from_leaf(depth, c));
        }
        depth[segment] = max_depth;
        return max_depth+1;
    }

    void depth_from_root(index_type& depth, int segment)
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

