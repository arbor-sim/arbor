#include "cell.hpp"

namespace nestmc {

cell::cell()
{
    // insert a placeholder segment for the soma
    segments_.push_back(make_segment<placeholder_segment>());
    parents_.push_back(0);
}

int cell::num_segments() const
{
    return segments_.size();
}

//
// note: I think that we have to enforce that the soma is the first
//       segment that is added
//
void cell::add_soma(value_type radius, point_type center)
{
    if(has_soma()) {
        throw std::domain_error(
            "attempt to add a soma to a cell that already has one"
        );
    }

    // add segment for the soma
    if(center.is_set()) {
        segments_[0] = make_segment<soma_segment>(radius, center);
    }
    else {
        segments_[0] = make_segment<soma_segment>(radius);
    }
}

void cell::add_cable(cell::index_type parent, segment_ptr&& cable)
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

segment* cell::segment(int index)
{
    if(index<0 || index>=num_segments()) {
        throw std::out_of_range(
            "attempt to access a segment with invalid index"
        );
    }
    return segments_[index].get();
}

segment const* cell::segment(int index) const
{
    if(index<0 || index>=num_segments()) {
        throw std::out_of_range(
            "attempt to access a segment with invalid index"
        );
    }
    return segments_[index].get();
}


bool cell::has_soma() const
{
    return !segment(0)->is_placeholder();
}

soma_segment* cell::soma()
{
    if(has_soma()) {
        return segment(0)->as_soma();
    }
    return nullptr;
}

cable_segment* cell::cable(int index)
{
    if(index>0 && index<num_segments()) {
        return segment(index)->as_cable();
    }
    return nullptr;
}

cell::value_type cell::volume() const
{
    return
       std::accumulate(
            segments_.begin(), segments_.end(),
            0.,
            [](double value, segment_ptr const& seg) {
                return seg->volume() + value;
            }
       );
}

cell::value_type cell::area() const
{
    return
       std::accumulate(
            segments_.begin(), segments_.end(),
            0.,
            [](double value, segment_ptr const& seg) {
                return seg->area() + value;
            }
       );
}

std::vector<segment_ptr> const& cell::segments() const
{
    return segments_;
}

void cell::construct() const
{
    if(num_segments()) {
        tree_ = cell_tree(parents_);
    }
}

cell_tree const& cell::graph() const
{
    construct();
    return tree_;
}

std::vector<int> const& cell::segment_parents() const
{
    return parents_;
}

} // namespace nestmc
