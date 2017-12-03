//#include <communication/global_policy.hpp>
//#include <domain_decomposition.hpp>
//#include <hardware/node_info.hpp>
//#include <recipe.hpp>


#include <vector>
#include <utility>

#include <common_types.hpp>
#include <util/debug.hpp>
#include <random>

#include <iostream>

namespace arb {

class connection_generator {
public:

    // Create
    connection_generator() {

    }

    // Returns a vector of pre-synaptic cell gids  for gid
    std::vector<cell_gid_type> connections(cell_gid_type gid
        //, rng
        ) const {
        std::vector<cell_gid_type> connections_on;
        // For now assume a single regular 2d grid with donut topology

        float gaussian_variance = 0.02;

        // number of neuron in population on a side
        cell_gid_type side = 100;
        cell_gid_type sources_per_cell = 100;


        // Sanity check
        EXPECTS(gid < (side * side));

        // Convert gid to location, then convert to between [0, 1)
        // x is the fast changing (columns)
        // y is rows
        float location_x = float(gid % side) / side;
        float location_y = float(gid / side) / side;

        // Generate the distribution for these locations
        std::normal_distribution<float> distr_x(location_x, gaussian_variance);
        std::normal_distribution<float> distr_y(location_y, gaussian_variance);

        //*********************************************************
        // now draw normal distributed and convert to gid
        // we have the amount of connections we need

        // Unique generator per cell, so should be supplied from the outside?
        // It should be reused for the other populations
        std::mt19937 generator_todo;

        for (cell_gid_type idx = 0; idx < sources_per_cell; ++idx) {
            // draw the number
            float x_source = distr_x(generator_todo);
            float y_source = distr_y(generator_todo);

            // normalize
            x_source -= int(x_source);
            y_source -= int(y_source);

            // convert to gid
            cell_gid_type gid_source = cell_gid_type(y_source * side) * side +
                cell_gid_type(x_source * side);
            connections_on.push_back(gid_source);

            std::cout << y_source << ", " << y_source << std::endl;
        }

        return connections_on;
    }


private:



};


}