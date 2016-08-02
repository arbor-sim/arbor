#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace nest {
namespace mc {

struct cell_count_info {
    cell_size_type num_sources;
    cell_size_type num_targets;
    cell_size_type num_probes;
};

class invalid_recipe_error: public std::runtime_error {
public:
    invalid_recipe_error(std::string whatstr): std::runtime_error(std::move(whatstr)) {}
};

/* Recipe descriptions are cell-oriented: in order that the building
 * phase can be done distributedly and in order that the recipe
 * description can be built indepdently of any runtime execution
 * environment, connection end-points are represented by pairs
 * (cell index, source/target index on cell).
 */

using cell_connection_endpoint = cell_member_type;

struct cell_connection {
    cell_connection_endpoint source;
    cell_connection_endpoint dest;

    float weight;
    float delay;
};

class recipe {
public:
    virtual cell_size_type num_cells() const =0;

    virtual cell get_cell(cell_gid_type) const =0; 
    virtual cell_count_info get_cell_count_info(cell_gid_type) const =0;
    virtual std::vector<cell_connection> connections_on(cell_gid_type) const =0;
};

} // namespace mc
} // namespace nest
