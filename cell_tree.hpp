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

class cell_tree {
    public :

    using int_type = int16_t;
    using index_type = memory::HostVector<int_type>;
    using index_view = index_type::view_type;

    cell_tree(std::vector<int> const& parent_index)
    {
        using range = memory::Range;

        // n = number of compartment in cell
        auto n = parent_index.size();

        // On completion of this loop child_count[i] is the number of children of compartment i
        // compensate count for compartment 0, which has itself as its own parent
        index_type child_count(n, 0);
        child_count[0] = -1;
        for(auto i : parent_index) {
            ++child_count[i];
        }

        // Find the number of branches by summing the number of children of all
        // compartments with more than 1 child.
        auto nbranches = std::accumulate(
            child_count.begin(), child_count.end(), 0,
            [](int total, int count) {return count > 1 ? total+count : total;}
        );
        nbranches++; // add an additional branch for the root/soma compartment

        // allocate space for storing branch data
        // note that the child data is generated after this loop
        data_   = index_type(4*nbranches);
        data_(memory::all) = 0; // initialize to zero
        depth_  = data_(0*nbranches, 1*nbranches);
        length_ = data_(1*nbranches, 2*nbranches);
        parent_ = data_(2*nbranches, 3*nbranches);
        end_    = data_(3*nbranches, 4*nbranches);

        // index of the branch for each compartment
        std::vector<int> branch_index(n);

        auto bcount=0;
        for(auto i : range(1,n)) {
            // index of the parent of compartment i
            auto parent_node = parent_index[i]; // the branch index of the parent of compartment i
            auto parent_branch = branch_index[parent_node];
            // reference to this compartment's parent index (to be determined)
            auto& this_branch = branch_index[i];

            // if this compartments's parent has more than one child, then this is the first
            // compartment in a branch, so mark it as such
            if(child_count[parent_node]>1) {
                bcount++;
                this_branch = bcount;
                parent_[bcount] = parent_branch;
                depth_[bcount]  = depth_[parent_branch]+1;
            }
            // not the first compartment in a branch, so inherit the parent's branch number
            else {
                this_branch = parent_branch;
            }
            length_[this_branch]++;
            end_[this_branch] = i;
        }

        // the number of children is the number of branches, excluding the root branch
        // num_children is equivalent to the number of edges in the graph
        auto num_children = nbranches-1;
        child_data_  = index_type(nbranches+1 + num_children, 0);
        child_index_ = child_data_(0, nbranches+1);
        children_    = child_data_(nbranches+1, memory::end);

        for(auto i : range(0, nbranches)) {
            // the number of children of a branch is the number of children of
            // the tail compartment in the branch
            auto c = child_count[end_[i]];
            child_index_[i+1] = c > 1 ? c : 0;
        }
        std::partial_sum(child_index_.begin(), child_index_.end(), child_index_.begin());

        for(auto i : range(1, nbranches)) {
            auto p = parent_[i];
            children_[child_index_[p]] = i;
            ++child_index_[p];
        }

        // use rotate to calculate indexes (after reverse iterator support is
        // available in the vector library!)
        //std::rotate(child_index_.rbegin(), child_index_.rbegin()+1,
        //            child_index_.rend());
        for(auto i=nbranches-1; i>0; --i) {
            child_index_[i+1] = child_index_[i];
        }
        child_index_[0] = 0;

        // Mark the parent of the root node as -1.
        // This simplifies the implementation of some algorithms on the tree
        // data structure.
        parent_[0] = -1;
    }

    /// memory used to store cell tree (in bytes)
    size_t memory() const {
        return   sizeof(int_type)*data_.size()
               + sizeof(int_type)*child_data_.size()
               + sizeof(cell_tree);
    }

    size_t num_branches() const {
        return child_index_.size()-1;
    }

    size_t num_children(size_t b) const {
        return child_index_[b+1] - child_index_[b];
    }

    const index_view children(size_t b) const {
        return children_(child_index_[b], child_index_[b+1]);
    }

    /// generate a graphviz .dot file for visualizing cell branch structure
    void to_graphviz(std::string const& fname) const {

        std::ofstream fid(fname);

        fid << "graph cell {" << std::endl;
        for(auto b : memory::Range(0,num_branches())) {
            if(children(b).size()) {
                for(auto c : children(b)) {
                    fid << "  " << b << " -- " << c << ";" << std::endl;
                }
            }
        }
        fid << "}" << std::endl;
    }

    /// update the depth of each branch to distance from leaf
    int_type depth_from_leaf() {
        return depth_from_leaf(0);
    }

    int_type balance() {
        if(num_branches()==1) {
            return 0;
        }

        auto max       = std::max_element(depth_.begin(), depth_.end());
        auto max_leaf  = std::distance(depth_.begin(), max);
        auto original_depth = *max;

        // Calculate the depth of each compartment as the maximum distance
        // from a child leaf
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
        auto pos = parent_[max_leaf];
        while(pos != -1) {
            for(auto c : children(pos)) {
                if(c!=parent) {
                    auto diameter = depth_[c] + 1 + distance_from_max_leaf;
                    if(diameter>max_sub_tree.diameter) {
                        max_sub_tree.set(pos, diameter, distance_from_max_leaf);
                    }
                }
            }
            parent = pos;
            pos = parent_[pos];
            ++distance_from_max_leaf;
        }

        std::cout << memory::util::green(std::to_string(max_sub_tree.root)) << " ";

        // calculate the depth of the balanced tree
        auto new_depth = (max_sub_tree.diameter+1) / 2;

        // nothing to do if the current root is also the root of the
        // balanced tree
        if(max_sub_tree.root==0 && max_sub_tree.depth==new_depth) {
            std::cout << " root " << 0 << " diameter " << original_depth << std::endl;
            return *max;
        }

        // perform another walk from max leaf towards max_sub_tree.root
        auto count = new_depth;
        auto new_root = max_leaf;
        while(count) {
            new_root = parent_[new_root];
            --count;
        }

        const auto root_children = children(new_root);
        auto diameter = *std::max_element(root_children.begin(), root_children.end());

        std::cout << " root " << new_root << " depth "
                  << original_depth << " -> " << new_depth  << std::endl;

        return new_depth;
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

    /// Recursive traversal of cell tree to determine ordering as maximum depth
    /// from a child leaf. Performs a post-order traversal.
    int_type depth_from_leaf(int branch) {
        int_type max_depth = 0;
        for(auto b : children(branch)) {
            max_depth = std::max(max_depth, depth_from_leaf(b));
        }
        depth_[branch] = max_depth;
        return max_depth+1;
    }


    index_type data_;

    // depth of the branch in cell tree
    index_view depth_;
    // number of compartments in branch
    index_view length_;
    // index of the parent compartment for this branch
    index_view parent_;
    // index of the last compartment in the branch
    index_view end_;

    // index of the last compartment in the branch
    index_type child_data_;

    index_view child_index_;
    index_view children_;
};

