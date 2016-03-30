#pragma once

#include <vector>
#include <stdexcept>

#include "segment.hpp"
#include "cell_tree.hpp"

namespace nestmc {

    // we probably need two cell types
    //  1. the abstract cell type (which would be this one)
    //  2.
    class cell {
        public:
        using index_type = int16_t;
        using value_type = double;
        using point_type = point<value_type>;

        int num_segments() const
        {
            return segments_.size();
        }

        void add_soma(value_type radius, point_type center=point_type())
        {
            if(has_soma()) {
                throw std::domain_error(
                    "attempt to add a soma to a cell that already has one"
                );
            }

            // soma has intself as its own parent
            soma_ = num_segments();
            parents_.push_back(num_segments());

            // add segment for the soma
            if(center.is_set()) {
                segments_.push_back(
                    make_segment<soma_segment>(radius, center)
                );
            }
            else {
                segments_.push_back(
                    make_segment<soma_segment>(radius)
                );
            }
        }

        void add_cable(segment_ptr&& cable, index_type parent)
        {
            // check for a valid parent id
            if(cable->is_soma()) {
                throw std::domain_error(
                    "attempt to add a soma as a segment"
                );
            }

            // check for a valid parent id
            if(parent>num_segments()) {
                throw std::out_of_range(
                    "parent index of cell segment is out of range"
                );
            }
            segments_.push_back(std::move(cable));
            parents_.push_back(parent);
        }

        template <typename... Args>
        void add_cable(index_type parent, Args ...args)
        {
            // check for a valid parent id
            if(parent>num_segments()) {
                throw std::out_of_range(
                    "parent index of cell segment is out of range"
                );
            }
            segments_.push_back(make_segment<cable_segment>(std::forward<Args>(args)...));
            parents_.push_back(parent);
        }

        bool has_soma() const { return soma_ > -1; }

        soma_segment* soma() {
            if(has_soma()) {
                return segments_[soma_].get()->as_soma();
            }
            return nullptr;
        }

        private:

        //
        // the local description of the cell which can be modified by the user
        // in a ad-hoc manner (adding segments, modifying segments, etc)
        //

        // storage for connections
        std::vector<index_type> parents_;
        // the segments
        std::vector<segment_ptr> segments_;
        // index of the soma
        int soma_ = -1;

        //
        // fixed cell description, which is computed from the
        // rough layout description above
        //

        // cell_tree that describes the connection layout
        //cell_tree tree_;
    };

} // namespace nestmc
