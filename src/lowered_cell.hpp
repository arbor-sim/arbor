#pragma once

namespace nest {
namespace mc {

template <typename Cell>
class lowered_cell {
    public :

    using cell_type = Cell;
    using value_type = typename cell_type::value_type;
    using size_type  = typename cell_type::value_type;

    private :

    cell_type cell_;
};

} // namespace mc
} // namespace nest
