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
        float stddev = 0.02;
        cell_gid_type count;

        // parameters for the synapses on this projection
        float weight_mean;
        float weight_std;

        float delay_min;        // Minimal delay
        float delay_per_std;    // per



        projection_pars(float var, cell_gid_type count, float weight_mean,
            float weight_std, float delay_min, float delay_per_std) :
            stddev(var), count(count),
            weight_mean(weight_mean), weight_std(weight_std),
            delay_min(delay_min), delay_per_std(delay_per_std){
            // Sanity checks
            EXPECTS(stddev > 0.0);
            EXPECTS(count > 0);
            EXPECTS(weight_mean > 0);
            EXPECTS(weight_std > 0);
            EXPECTS(delay_min > 1.0); // TODO: This a neuroscientific 'fact' not needed for valid functioning
            EXPECTS(delay_per_std > 0.0);
        }
    };

    struct synaps_pars {
        cell_gid_type gid;
        float weight;
        float delay;
        synaps_pars(cell_gid_type gid, float weight, float delay ):
             gid(gid),  weight(weight), delay(delay){}
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
    std::vector<synaps_pars> synapses_on(cell_gid_type gid) {
        std::mt19937 gen;
        gen.seed(gid);

        std::vector<synaps_pars> connections;
        for (auto projection : connectome_)
        {
            // Sanity check that the populations exist
            EXPECTS(std::get<0>(projection) < populations_.size());
            EXPECTS(std::get<1>(projection) < populations_.size());

            // Shorthand for the pre and post populations
            auto pre_pop = populations_[std::get<0>(projection)];
            auto post_pop = populations_[std::get<1>(projection)];
            auto pro_pars = std::get<2>(projection);

            // Distribution to draw the weights
            std::normal_distribution<float> weight_distr(pro_pars.weight_mean, pro_pars.weight_std);
            float weight_sign = pro_pars.weight_mean < 0 ? -1 : 1;
            // Check if this gid receives connections via this
            // projection
            // TODO: Replace with the fance in range function we have somewhere in the utils
            if (gid < post_pop.start_index || gid > (post_pop.start_index + post_pop.n_cells)) {
                continue;
            }

            // We need the local gid of the post neuron
            // TODO: is this check needed?
            auto pop_local_gid = gid - post_pop.start_index;
            EXPECTS(pop_local_gid < (post_pop.x_side * post_pop.y_side));

            // Convert this to a normalized location
            std::pair<float, float> post_location = {
                float(pop_local_gid % post_pop.x_side) / post_pop.x_side,
                float(pop_local_gid / post_pop.y_side) / post_pop.y_side};

            // Now we sample from the pre population based on the x,y location
            // We supply the connection_impl with the size of the pre polulation
            auto pre_locations = connection_locations(gen, post_location,
                pro_pars, post_pop.periodic);

            for (auto pre_location : pre_locations)
            {
                // convert the normalized locations to gid
                cell_gid_type gid_pre = cell_gid_type(pre_location.second * pre_pop.y_side) * pre_pop.x_side +
                    cell_gid_type(pre_location.first * pre_pop.x_side);

                // Calculate the distance between the pre and post neuron. use
                float distance = std::sqrt(std::pow(pre_location.first * post_location.first, 2) +
                    std::pow(pre_location.second * post_location.second, 2));

                float delay = distance / pro_pars.delay_per_std + pro_pars.delay_min;

                float weight = weight_distr(gen);
                // Flip the sign of the weight depending if we are incorrect
                // depending oninhib or exit
                weight = (weight_sign * weight) < 0?  -weight: weight;

                //std::cout << x_source << ", -" << int(x_source) << " = " << x_source << "    |    "
                //    << y_source << " -" << int(y_source) << " = " << x_source
                //    << " : " << gid_source << std::endl;
                connections.push_back({ gid_pre ,weight,  delay });
            }
        }

        return connections;
    }

    std::vector<std::pair<float, float>> connection_locations(std::mt19937 gen,
        std::pair<float, float> const target_location, projection_pars const pars, bool periodic)
    {
        // Generate the distribution for these locations
        std::normal_distribution<float> distr_x(target_location.first, pars.stddev);
        std::normal_distribution<float> distr_y(target_location.second, pars.stddev);

        //*********************************************************
        // now draw normal distributed and convert to gid
        // we have the amount of connections we need
        std::vector<std::pair<float, float>> connections;

        for (cell_gid_type idx = 0; idx < pars.count; ++idx) {
            // draw the locations
            float x_source = distr_x(gen);
            float y_source = distr_y(gen);

            if (periodic)
            {
                // Todo: add non-periodic borders
                // normalize: move all values between [0.0, 1.0)
                // int(floor(-1.1)) = -2  ---> -1.1 - -2 = 0.9
                // int(floor(3.4)) = 3    ---> 3.4  -  3 = 0.4
                x_source -= int(std::floor(x_source));
                y_source -= int(std::floor(y_source));
            }
            else {
                // If we have non periodic borders this connection is not
                // created (akin to a in vitro slice) if outside of [0, 1.0)
                if (x_source < 0 || x_source >= 1.0 || y_source < 0 || y_source >= 1.0) {
                    continue;
                }
            }
            connections.push_back( { x_source , y_source });
        }

        return connections;
    }

private:
    std::vector<population> populations_;
    std::vector<std::tuple<unsigned, unsigned, projection_pars>> connectome_;


};


}