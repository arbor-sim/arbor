#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace nest {
namespace mc {

using cell_id_type = std::size_t;
using cell_size_type = std::size_t;

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

struct cell_connection_endpoint {
    cell_id_type cell;
    unsigned endpoint_index;
};

struct cell_connection {
    cell_connection_endpoint source;
    cell_connection_endpoint dest;

    float weight;
    float delay;
};

class recipe {
public:
    virtual cell_id_type num_cells() const =0;

    virtual cell get_cell(cell_id_type) const =0; 
    virtual cell_count_info get_cell_count_info(cell_id_type) const =0;
    virtual std::vector<cell_connection> connections_on(cell_id_type) const =0;
};

// miniapp-specific recipes

struct probe_distribution {
    float proportion = 1.f; // what proportion of cells should get probes?
    bool all_segments = true;    // false => soma only
    bool membrane_voltage = true;
    bool membrane_current = true;
};

struct basic_recipe_param {
    unsigned num_compartments = 1;
    unsigned num_synapses = 1;
    std::string synapse_type = "expsyn";
    float min_connection_delay_ms = 20.0;
    float mean_connection_delay_ms = 20.75;
    float syn_weight_per_cell = 0.3;
};

std::unique_ptr<recipe> make_basic_ring_recipe(
        cell_id_type ncell,
        basic_recipe_param param,
        probe_distribution pdist = probe_distribution{});

std::unique_ptr<recipe> make_basic_kgraph_recipe(
        cell_id_type ncell,
        basic_recipe_param param,
        probe_distribution pdist = probe_distribution{});

std::unique_ptr<recipe> make_basic_rgraph_recipe(
        cell_id_type ncell,
        basic_recipe_param param,
        cell_count_type cell_fan_in,
        probe_distribution pdist = probe_distribution{});

} // namespace mc
} // namespace nest
