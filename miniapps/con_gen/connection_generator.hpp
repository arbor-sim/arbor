//#include <domain_decomposition.hpp>
//#include <hardware/node_info.hpp>
//#include <recipe.hpp>

#pragma once

#include <vector>
#include <utility>
#include <tuple>
#include <math.h>

#include <common_types.hpp>
#include <util/debug.hpp>
#include <random>

#include <iostream>

namespace arb {

    // Describes a 2d surface of neurons located on grid locations
    // -x_side   number of neurons on the x-side
    // -y_side   number of neurons on the y-side
    // -periodic Do the border loop back to the other side (torus topology)
    struct population {
    public:
        cell_gid_type x_side;
        cell_gid_type y_side;
        bool periodic;

        cell_gid_type n_cells;

        population(cell_gid_type x_side, cell_gid_type y_side, bool per) :
            x_side(x_side), y_side(y_side), periodic(per), n_cells(x_side *y_side) {

            // Sanity check
            EXPECTS(x_side > 0);
            EXPECTS(y_side > 0);
        }
    };

    // Describes a projection between the neurons between two populations
    // - sd      sd of the normal distributed used to sample the pre_synaptic
    //           The dimensions of the pre-population is sampled as if it has size 1.0 * 1.0
    // - count   Number of samples to take. When sampling from a non periodic population
    //           this count can be lower (akin with a sample in-vitro)
    //
    // - weight_mean  Mean synaptic weight for the created synapse
    // - weight_sd    Standard deviation around mean for sampling the weights
    //
    // - delay_min      Minimal delay of the created synapse
    // - delay_per_sd   Delay increase by sd distance between neurons
    struct projection_pars {
        float sd = 0.02;
        cell_gid_type count;

        // parameters for the synapses on this projection
        float weight_mean;
        float weight_sd;

        float delay_min;        // Minimal delay
        float delay_per_sd;    // per



        projection_pars(float var, cell_gid_type count, float weight_mean,
            float weight_std, float delay_min, float delay_per_std) :
            sd(var), count(count),
            weight_mean(weight_mean), weight_sd(weight_std),
            delay_min(delay_min), delay_per_sd(delay_per_std){
            // Sanity checks
            EXPECTS(sd > 0.0);
            EXPECTS(count > 0);
            EXPECTS(weight_mean > 0);
            EXPECTS(weight_std > 0);
            EXPECTS(delay_min > 1.0); // TODO: This a neuroscientific 'fact' not needed for valid functioning
            EXPECTS(delay_per_std > 0.0);
        }
    };

    // Helper struct to collect some parameters together
    // -pre_idx    The index in the population list that is pre synaptic for this
    //             projection
    // -post_idx   The index in the population list that is post synaptic for this
    //             projection
    // -pars       Parameters used to generate the synapses for this connection
    struct projection {
        unsigned pre_idx;
        unsigned post_idx;
        projection_pars pars;

        projection(unsigned pre_population, unsigned post_population, projection_pars pars) :
            pre_idx(pre_population), post_idx(post_population), pars(pars) {}
    };

    // Return type for connection generation
    // A set of pre-synaptic cell gid,
    // weight and delay
    struct synaps_pars {
        cell_gid_type gid;
        float weight;
        float delay;
        synaps_pars(cell_gid_type gid, float weight, float delay ):
             gid(gid),  weight(weight), delay(delay){}
    };


class connection_generator {

public:
    // Connection generator.
    // Expects a vector of populations descriptions and vector of projections
    // between these
    // TODO: This is a first implementation: sub populations are NOT implemented
    connection_generator( std::vector<population> const populations,
        std::vector<projection> const connectome)
        : connectome_(connectome) {
        cell_gid_type gid_idx = 0;

        // Create the local populations with start index set
        for (auto pop : populations) {
            populations_.push_back(population_indexed(
                pop.x_side, pop.y_side, pop.periodic, gid_idx ));

            gid_idx += pop.n_cells;
        }
    }

    // Returns a vector of all synaptic parameters sets for this gid
    std::vector<synaps_pars> synapses_on(cell_gid_type gid) {
        std::mt19937 gen;
        gen.seed(gid);

        std::vector<synaps_pars> connections;
        for (auto project : connectome_)
        {
            // Sanity check that the populations exist
            EXPECTS(project.pre_idx < populations_.size());
            EXPECTS(project.post_idx < populations_.size());

            // Shorthand for the pre and post populations
            auto pre_pop = populations_[project.pre_idx];
            auto post_pop = populations_[project.post_idx];
            auto pro_pars = project.pars;

            // Distribution to draw the weights
            std::normal_distribution<float> weight_distr(pro_pars.weight_mean, pro_pars.weight_sd);

            // Used to assure correct weight sign
            float weight_sign = pro_pars.weight_mean < 0 ? -1 : 1;

            // Check if this gid receives connections via this projection
            // TODO: Replace with the fancy in range function we have somewhere in the utils
            if (gid < post_pop.start_index || gid > (post_pop.start_index + post_pop.n_cells)) {
                continue;
            }

            // Convert to the local gid of the post neuron
            auto pop_local_gid = gid - post_pop.start_index;

            // Convert this to a normalized location
            std::pair<float, float> post_location = {
                float(pop_local_gid % post_pop.x_side) / post_pop.x_side,
                float(pop_local_gid / post_pop.y_side) / post_pop.y_side};

            // If we have non square sides we need to correct the stdev to get
            // circular projections!
            float sd_x = pro_pars.sd;
            float sd_y = pro_pars.sd;
            if (post_pop.x_side != post_pop.y_side) {
                if (post_pop.x_side < post_pop.y_side) {
                    float ratio = float(post_pop.y_side) / post_pop.x_side;
                    sd_x *= ratio;
                }
                else {
                    float ratio = float(post_pop.x_side) / post_pop.y_side;
                    sd_y *= ratio;
                }
            }

            // Now we sample from the pre population based on the x,y location
            // We supply the connection_impl with the size of the pre population
            auto pre_locations = connection_locations(gen, post_location,
                pro_pars.count, sd_x, sd_y, post_pop.periodic);

            for (auto pre_location : pre_locations)
            {
                // convert the normalized locations to gid
                cell_gid_type gid_pre = cell_gid_type(pre_location.second * pre_pop.y_side) * pre_pop.x_side +
                    cell_gid_type(pre_location.first * pre_pop.x_side);

                // Calculate the distance between the pre and post neuron.
                float distance = std::sqrt(std::pow(pre_location.first * post_location.first, 2) +
                    std::pow(pre_location.second * post_location.second, 2));

                float delay = distance / pro_pars.delay_per_sd + pro_pars.delay_min;

                float weight = weight_distr(gen);
                // Flip the sign of the weight depending if we are incorrect
                weight = (weight_sign * weight) < 0?  -weight: weight;

                connections.push_back({ gid_pre + pre_pop.start_index, weight,  delay });
            }
        }

        return connections;
    }

    std::vector<std::pair<float, float>> connection_locations(std::mt19937 gen,
        std::pair<float, float> const target_location, unsigned count,
        float sd_x, float sd_y, bool periodic)
    {
        // Generate the distribution for these locations
        std::normal_distribution<float> distr_x(target_location.first, sd_x);
        std::normal_distribution<float> distr_y(target_location.second, sd_y);

        //*********************************************************
        // now draw normal distributed and convert to gid
        // we have the amount of connections we need
        std::vector<std::pair<float, float>> connections;

        for (cell_gid_type idx = 0; idx < count; ++idx) {
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
    struct population_indexed : public population {
        cell_gid_type start_index;

        population_indexed(cell_gid_type x_side, cell_gid_type y_side, bool periodic,
            cell_gid_type start_index) :
            population(x_side, y_side, periodic), start_index(start_index) {}
    };

    std::vector<population_indexed> populations_;
    std::vector<projection> connectome_;
};

}