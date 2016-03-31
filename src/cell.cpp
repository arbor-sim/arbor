#include "cell.hpp"

namespace nestmc {

int cell::num_segments() const
{
    return segments_.size();
}

void cell::add_soma(value_type radius, point_type center)
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

// add a cable that is provided by the user as a segment_ptr
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


bool cell::has_soma() const
{
    return soma_ > -1;
}

soma_segment* cell::soma()
{
    if(has_soma()) {
        return segments_[soma_].get()->as_soma();
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

void cell::construct()
{
    if(num_segments()) {
        tree_ = cell_tree(parents_);
    }
}

cell_tree const& cell::graph() const
{
    return tree_;
}

std::vector<int> const& cell::segment_parents() const
{
    return parents_;
}

} // namespace nestmc
