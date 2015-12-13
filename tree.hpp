#pragma once

#include <algorithm>
#include <vector>

#include "vector/include/Vector.hpp"
#include "util.hpp"

using range = memory::Range;

class tree {
    public :

    using int_type = int16_t;

    using index_type = memory::HostVector<int_type>;
    using index_view = index_type::view_type;

    tree(tree const& other)
    :   data_(other.data_)
    {
        set_ranges(other.num_nodes());
    }

    tree(tree&& other)
    {
        data_ = std::move(other.data_);
        set_ranges(other.num_nodes());
    }

    tree() = default;

    /// create the tree from a parent_index
    /// maybe this should be an initializer, not a constructor, because
    /// we currently return the branch index in place of the parent_index
    /// which feels like bad design.
    template <typename I>
    std::vector<I>
    init_from_parent_index(std::vector<I> const& parent_index)
    {
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
        auto nbranches = 1 + child_count[0]; // children of the root node are counted differently
        for(auto i : range(1,n)) {
            if(child_count[i]>1) {
                nbranches += child_count[i];
            }
        }

        // The number of children is the number of branches, excluding the root branch.
        // num_children is equivalent to the number of edges in the graph.
        auto nchildren = nbranches-1;

        // allocate memory for storing the tree
        init(nbranches);

        // index of the branch for each compartment
        std::vector<I> branch_index(n);

        // Mark the parent of the root node as -1.
        // This simplifies the implementation of some algorithms on the tree
        parents_[0]=-1;

        // bcount records how many branches have been created in the loop
        auto bcount=0;

        for(auto i : range(1,n)) {
            // the branch index of the parent of compartment i
            auto parent_node = parent_index[i];
            // index of the parent of compartment i
            auto parent_branch = branch_index[parent_node];

            // if this compartments's parent
            //  - has more than one child
            //  - or is the root branch
            // this is the first compartment in a branch, so mark it as such
            if(child_count[parent_node]>1 || parent_node==0) {
                bcount++;
                branch_index[i] = bcount;
                parents_[bcount] = parent_branch;
            }
            // not the first compartment in a branch,
            // so inherit the parent's branch number
            else {
                branch_index[i] = parent_branch;
            }
        }

        child_index_(memory::all) = 0;
        // the root node has to be handled separately: all of its children must
        // be counted
        child_index_[1] = child_count[0];
        for(auto i : range(1, n)) {
            if(child_count[i]>1) {
                child_index_[branch_index[i]+1] = child_count[i];
            }
        }
        std::partial_sum(child_index_.begin(), child_index_.end(), child_index_.begin());

        // Fill in the list of children of each branch.
        // Requires some additional book keeping to keep track of how many
        // children have already been filled in for each branch.
        for(auto i : range(1, nbranches)) {
            // parents_[i] is the parent of branch i, for which i must be added
            // as a child, and child_index_[p] is the index into which the next
            // child of p is to be stored
            auto p = parents_[i];
            children_[child_index_[p]] = i;
            ++child_index_[p];
        }

        // The child index has already been calculated as a side-effect of the
        // loop above, but is shifted one value to the left, so perform a
        // rotation to the right.
        for(auto i=nbranches-1; i>=00; --i) {
            child_index_[i+1] = child_index_[i];
        }
        child_index_[0] = 0;

        // return the branch index to the caller for later use
        return branch_index;
    }

    int num_children() const {
        return children_.size();
    }
    int num_children(int b) const {
        return child_index_[b+1] - child_index_[b];
    }
    int num_nodes() const {
        return child_index_.size() - 1;
    }

    /// return the child index
    const index_view child_index() const {
        return child_index_;
    }

    /// return the list of all children
    const index_view children() const {
        return children_;
    }

    /// return the list of all children of branch b
    const index_view children(int b) const {
        return children_(child_index_[b], child_index_[b+1]);
    }

    /// return the list of parents
    const index_view parents() const {
        return parents_;
    }

    /// return the parent of branch b
    int_type parent(int b) const {
        return parents_[b];
    }
    int_type& parent(int b) {
        return parents_[b];
    }

    /// memory used to store tree (in bytes)
    size_t memory() const {
        return sizeof(int_type)*data_.size() + sizeof(tree);
    }

    private :

    void init(int nnode) {
        auto nchild = nnode -1;

        data_ = index_type(nchild + (nnode + 1) + nnode);
        set_ranges(nnode);
    }

    void set_ranges(int nnode) {
        auto nchild = nnode - 1;
        // data_ is partitioned as follows:
        // data_ = [children_[nchild], child_index_[nnode+1], parents_[nnode]]
        assert(data_.size() == nchild + (nnode+1) + nnode);
        children_    = data_(0, nchild);
        child_index_ = data_(nchild, nchild+nnode+1);
        parents_     = data_(nchild+nnode+1, memory::end);

        // check that arrays have appropriate size
        // this should be moved into a unit test
        assert(children_.size()    == nchild);
        assert(child_index_.size() == nnode+1);
        assert(parents_.size()     == nnode);
    }

    /// Renumber the sub-tree with old_branch as its root with new_branch as
    /// the new index of old_branch. This is a helper function for the
    /// renumbering required when a new root node is chosen to improve
    /// the balance of the tree.
    /// Optionally add the parent of old_branch as a child of new_branch, which
    /// will be applied recursively until the old root has been processed,
    /// which indicates that the renumbering is finished.
    /*
    int add_child( int new_branch, int old_branch, index_view p,
                   bool parent_as_child)
    {
        // check for the senitel that indicates that the old root has
        // been processed
        if(old_branch==-1) {
            assert(parent_as_child); // sanity check
            return new_branch;
        }

        auto kids = children(old_branch);
        auto pos = new_child_index[new_branch];

        new_child_index[new_branch+1] = pos + kids.size();
        // add an extra one if adding the parent as a child
        if(parent_as_child) {
            ++new_child_index[new_branch+1];
        }

        // first renumber the children
        for(auto b : kids) {
            p[new_branch] = b;
            new_children[pos++] = new_branch++;
        }
        // then add and renumber the parent as a child
        if(parent_as_child) {
            p[new_branch] = parents_[old_branch];
            new_children[pos++] = new_branch++;
        }

        // then visit the sub-tree of each child recursively
        //      - traverse _down_ the tree
        for(auto b : kids) {
            new_branch = add_children(new_branch, b, p, false);
        }
        // finally visit the parent recursively
        //      - traverse _up_ the tree towards the old root
        if(parent_as_child) {
            new_branch = add_children(new_branch, parents_[old_branch], p, true);
        }

        return new_branch;
    }
    */

    index_type data_;
    index_view children_;
    index_view child_index_;
    index_view parents_;
};
