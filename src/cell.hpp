#pragma once

#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "segment.hpp"
#include "cell_tree.hpp"

namespace nestmc {

    // high-level abstract representation of a cell and its segments
    class cell {
        public:

        // types
        using index_type = int;
        using value_type = double;
        using point_type = point<value_type>;

        // constructor
        cell();

        /// add a soma to the cell
        /// radius must be specified
        void add_soma(value_type radius, point_type center=point_type());

        /// add a cable
        /// parent is the index of the parent segment for the cable section
        /// cable is the segment that will be moved into the cell
        void add_cable(index_type parent, segment_ptr&& cable);

        /// add a cable by constructing it in place
        /// parent is the index of the parent segment for the cable section
        /// args are the arguments to be used to consruct the new cable
        template <typename... Args>
        void add_cable(index_type parent, Args ...args);

        /// the number of segments in the cell
        int num_segments() const;

        bool has_soma() const;

        class segment* segment(int index);
        class segment const* segment(int index) const;

        /// access pointer to the soma
        /// returns nullptr if the cell has no soma
        soma_segment* soma();

        /// access pointer to a cable segment
        /// will throw an std::out_of_range exception if
        /// the cable index is not valid
        cable_segment* cable(int index);

        /// the volume of the cell
        value_type volume() const;

        /// the surface area of the cell
        value_type area() const;

        std::vector<segment_ptr> const& segments() const;

        /// the connectivity graph for the cell segments
        cell_tree const& graph() const;

        /// return reference to array that enumerates the index of the parent of
        /// each segment
        std::vector<int> const& segment_parents() const;

        private:

        /// generate the internal representation of the connectivity
        /// graph for the cell segments
        void construct() const;


        //
        // the local description of the cell which can be modified by the user
        // in a ad-hoc manner (adding segments, modifying segments, etc)
        //

        // storage for connections
        std::vector<index_type> parents_;
        // the segments
        std::vector<segment_ptr> segments_;

        //
        // fixed cell description, which is computed from the layout description
        // this computed whenever a call to the graph() method is made
        // the graph method is const, so tree_ is mutable
        //
        // note: this isn't remotely thread safe!

        mutable cell_tree tree_;
    };

    // create a cable by forwarding cable construction parameters provided by the user
    template <typename... Args>
    void cell::add_cable(cell::index_type parent, Args ...args)
    {
        // check for a valid parent id
        if(parent>=num_segments()) {
            throw std::out_of_range(
                "parent index of cell segment is out of range"
            );
        }
        segments_.push_back(make_segment<cable_segment>(std::forward<Args>(args)...));
        parents_.push_back(parent);
    }


} // namespace nestmc

