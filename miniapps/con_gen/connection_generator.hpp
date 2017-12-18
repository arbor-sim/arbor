//#include <domain_decomposition.hpp>
//#include <hardware/node_info.hpp>
//#include <recipe.hpp>

#pragma once

#include <vector>
#include <utility>
#include <tuple>
#include <math.h>


#include <common_types.hpp>
#include <math.hpp>
#include <random>
#include <util/debug.hpp>

namespace arb_con_gen {

    using namespace arb;

    // Describes a 2d surface of neurons located on grid locations
    // -x_dim   number of neurons on the x-side
    // -y_dim   number of neurons on the y-side
    // -periodic Do the border loop back to the other side (torus topology)
    struct population {
        cell_size_type x_dim;
        cell_size_type y_dim;
        bool periodic;

        cell_size_type n_cells;

        // TODO: enum topology_type ( grid, pure random, minimal distance)

        population(cell_size_type x_dim, cell_size_type y_dim, bool per) :
            x_dim(x_dim), y_dim(y_dim), periodic(per), n_cells(x_dim *y_dim)
        {

            // Sanity check
            EXPECTS(x_dim > 0);
            EXPECTS(y_dim > 0);
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
        cell_size_type count;

        // parameters for the synapses on this projection
        float weight_mean;
        float weight_sd;

        float delay_min;        // Minimal delay
        float delay_per_sd;    // per

        projection_pars(float var, cell_size_type count, float weight_mean,
            float weight_std, float delay_min, float delay_per_std) :
            sd(var), count(count),
            weight_mean(weight_mean), weight_sd(weight_std),
            delay_min(delay_min), delay_per_sd(delay_per_std)
        {
            // Sanity checks
            EXPECTS(sd > 0.0);
            EXPECTS(count > 0);
            EXPECTS(weight_mean > 0);
            EXPECTS(weight_std > 0);
            EXPECTS(delay_min > 1.0); // TODO: This a neuroscientific 'fact' not needed for valid functioning
            EXPECTS(delay_per_std > 0.0);
        }
    };

    // Helper struct to collect some parameters together for creating projections
    // -pre_idx    The index in the population list that is pre synaptic for this
    //             projection
    // -post_idx   The index in the population list that is post synaptic for this
    //             projection
    // -pars       Parameters used to generate the synapses for this connection
    struct projection {
        size_t pre_idx;
        size_t post_idx;
        projection_pars pars;

        projection(size_t pre_population, size_t post_population, projection_pars pars) :
            pre_idx(pre_population), post_idx(post_population), pars(pars)
        {}
    };

    // Return type for connection generation
    // A set of pre-synaptic cell gid,
    // weight and delay
    struct synaps_pars {
        cell_gid_type gid;
        float weight;
        float delay;
        synaps_pars(cell_gid_type gid, float weight, float delay ):
             gid(gid),  weight(weight), delay(delay)
        {}
    };

class connection_generator {

public:
    // Connection generator.
    // Expects a vector of populations descriptions and vector of projections
    // between these
    // TODO: This is a first implementation: sub populations are NOT implemented
    connection_generator(const std::vector<population> & populations,
        std::vector<projection> connectome):
        connectome_(std::move(connectome))
    {
        cell_gid_type gid_idx = 0;

        // Create the local populations with start index set
        for (auto pop : populations) {
            populations_.push_back(population_indexed(
                pop.x_dim, pop.y_dim, pop.periodic, gid_idx ));

            gid_idx += pop.n_cells;
        }
    }

    // Returns a vector of all synaptic parameters sets for this gid
    std::vector<synaps_pars> synapses_on(cell_gid_type gid) {
        std::mt19937 gen;
        gen.seed(gid);

        std::vector<synaps_pars> connections;
        for (auto project : connectome_) {

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
            float weight_sign = arb::math::signum(pro_pars.weight_mean);

            // Check if this gid receives connections via this projection
            // TODO: Replace with the fancy in range function we have somewhere in the utils
            if (gid < post_pop.start_index || gid > (post_pop.start_index + post_pop.n_cells)) {
                continue;
            }

            // Convert to the local gid of the post neuron
            auto pop_local_gid = gid - post_pop.start_index;

            // Convert this to a normalized location
            point post_location = {
                float(pop_local_gid % post_pop.x_dim) / post_pop.x_dim,
                float(pop_local_gid / post_pop.y_dim) / post_pop.y_dim};

            // If we have non square sides we need to correct the stdev to get
            // circular projections!
            float sd_x = pro_pars.sd;
            float sd_y = pro_pars.sd;
            if (post_pop.x_dim != post_pop.y_dim) {
                if (post_pop.x_dim < post_pop.y_dim) {
                    float ratio = float(post_pop.y_dim) / post_pop.x_dim;
                    sd_x *= ratio;
                }
                else {
                    float ratio = float(post_pop.x_dim) / post_pop.y_dim;
                    sd_y *= ratio;
                }
            }

            // Now we sample from the pre population based on the x,y location of the
            // post cell
            auto pre_locations = get_random_locations(gen, post_location,
                pro_pars.count, sd_x, sd_y, post_pop.periodic);

            // Convert to gid and draw the synaptic parameters for each
            // generated location
            for (auto pre_location : pre_locations) {

                // If we have Grid type topology
                // convert the normalized locations to gid
                cell_gid_type gid_pre = cell_gid_type(pre_location.y * pre_pop.y_dim) * pre_pop.x_dim +
                    cell_gid_type(pre_location.x * pre_pop.x_dim);
                // absolute gid
                gid_pre += pre_pop.start_index;

                // TODO: If we have randomly distributed cell, use a quadtree to find the gid

                // Calculate the distance between the pre and post neuron.
                float distance = std::sqrt(std::pow(pre_location.x * post_location.x, 2) +
                    std::pow(pre_location.y * post_location.y, 2));

                float delay = distance / pro_pars.delay_per_sd + pro_pars.delay_min;

                float weight = weight_distr(gen);
                // Flip the sign of the weight depending if we are incorrect
                weight = (weight_sign * weight) < 0?  -weight: weight;

                connections.push_back({ gid_pre, weight,  delay });
            }
        }

        return connections;
    }

private:

    struct point {
        float x;
        float y;
    };


    // Returns a vector of points from a 2d normal distribution around the
    // supplied 2d location.
    std::vector<point> get_random_locations(std::mt19937 gen,
        point target_location, cell_size_type count,
        float sd_x, float sd_y, bool periodic)
    {
        // Generate the distribution for these locations
        std::normal_distribution<float> distr_x(target_location.x, sd_x);
        std::normal_distribution<float> distr_y(target_location.y, sd_y);

        //*********************************************************
        // now draw normal distributed and convert to gid
        // we have the amount of connections we need
        std::vector<point> connections;

        for (cell_gid_type idx = 0; idx < count; ++idx) {

            // draw the locations
            float x_source = distr_x(gen);
            float y_source = distr_y(gen);

            if (periodic) {
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
            connections.push_back({ x_source , y_source });
        }

        return connections;
    }

    float distance(const point& p1, const point& p2) {
        return std::sqrt(arb::math::square(p1.x - p2.x) + arb::math::square(p1.y - p2.y));
    }


    struct population_indexed : public population {
        cell_gid_type start_index;

        population_indexed(cell_size_type x_dim, cell_size_type y_dim, bool periodic,
            cell_gid_type start_index) :
            population(x_dim, y_dim, periodic), start_index(start_index)
        {}
    };

    std::vector<population_indexed> populations_;
    std::vector<projection> connectome_;
};

}