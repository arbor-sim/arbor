//#include <domain_decomposition.hpp>
//#include <hardware/node_info.hpp>
//#include <recipe.hpp>


#include <vector>
#include <utility>
#include <tuple>
#include <math.h>

#include <common_types.hpp>
#include <util/debug.hpp>
#include <random>

#include <iostream>

namespace arb {

    struct population {

    public:
        cell_gid_type x_side;
        cell_gid_type y_side;
        bool periodic;

        // Maybee have a private struct that has this start index?
        cell_gid_type start_index;

        cell_gid_type n_cells;

        population(cell_gid_type x_side, cell_gid_type y_side, bool per,
            cell_gid_type start_index = 0) :
            x_side(x_side), y_side(y_side), periodic(per),
            start_index(start_index), n_cells(x_side *y_side) {

            // Sanity check
            EXPECTS(x_side > 0);
            EXPECTS(y_side > 0);
        }
    };

    struct projection_pars {
        float var = 0.02;
        cell_gid_type count;

        projection_pars(float var, cell_gid_type count) :
            var(var), count(count) {}
    };

class connection_generator {
public:



    // Create
    connection_generator( std::vector<population> const populations,
        std::vector<std::tuple<unsigned, unsigned, projection_pars>> const connectome)
        : connectome_(connectome) {
        cell_gid_type gid_idx = 0;
        // Create the local (instantiated) populations with start index set
        for (auto pop : populations) {
            populations_.push_back(population(pop.x_side, pop.y_side, pop.periodic, gid_idx));
            gid_idx += pop.n_cells;
        }
    }

    // Returns a vector of pre-synaptic cell gids  for gid
    std::vector<cell_gid_type> pre_synaptic_cells(cell_gid_type gid) {
        std::mt19937 gen;
        gen.seed(gid);

        std::vector<cell_gid_type> connections;
        for (auto projection : connectome_)
        {
            std::cout << "debug 1" << std::endl;
            // Sanity check that the populations exist
            EXPECTS(std::get<0>(projection) < populations_.size());
            EXPECTS(std::get<1>(projection) < populations_.size());

            // Shorthand for the pre and post populations
            auto pre_pop = populations_[std::get<0>(projection)];
            auto post_pop = populations_[std::get<1>(projection)];

            // Check if this gid receives connections via this
            // projection
            // TODO: Replace with the fance in range function we have somewhere in the utils
            if (gid < post_pop.start_index || gid > (post_pop.start_index + post_pop.n_cells)) {

                std::cout << gid << ", " << post_pop.start_index
                    << post_pop.start_index << ", " << post_pop.n_cells << std::endl;
                continue;
            }
            std::cout << "debug 2" << std::endl;
            // From the projection get the parameters and generate the gids
            auto pro_pars = std::get<2>(projection);
            // abs GID to local gid by substracting start index in population
            auto population_cells = connections_impl(gen, gid - post_pop.start_index,
                pro_pars.var, pro_pars.count, post_pop.x_side, post_pop.y_side);

            // Collect all together
            connections.insert(connections.end(),
                population_cells.begin(), population_cells.end());

            std::cout << "debug 3: " << connections.size() << std::endl;
        }

        return connections;
    }





    std::vector<cell_gid_type> connections_impl(std::mt19937 gen, cell_gid_type gid,
        float var, cell_gid_type count, cell_gid_type x_side, cell_gid_type y_side )
    {
        // Sanity check
        EXPECTS(gid < (x_side * y_side));

        // Convert gid to location, then convert to between [0, 1)
        // x is the fast changing (columns)
        // y is rows
        float location_x = float(gid % x_side) / x_side;
        float location_y = float(gid / x_side) / y_side;


        // Generate the distribution for these locations
        std::normal_distribution<float> distr_x(location_x, var);
        std::normal_distribution<float> distr_y(location_y, var);

        //*********************************************************
        // now draw normal distributed and convert to gid
        // we have the amount of connections we need
        std::vector<cell_gid_type> connections;

        for (cell_gid_type idx = 0; idx < count; ++idx) {
            // draw the locations
            float x_source = distr_x(gen);
            float y_source = distr_y(gen);

            // normalize
            x_source -= int(x_source);
            y_source -= int(y_source);

            std::cout << x_source << ", " << y_source << std::endl;
            // convert to gid
            cell_gid_type gid_source = cell_gid_type(y_source * y_side) * x_side +
                cell_gid_type(x_source * x_side);
            connections.push_back(gid_source);
        }

        return connections;
    }

private:
    std::vector<population> populations_;
    std::vector<std::tuple<unsigned, unsigned, projection_pars>> connectome_;


};


}