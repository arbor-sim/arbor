/* First go at refactoring miniapp model and model descriptions.
 * Thoughts:
 *
 *   recipe:   a description of a network, ideally declarative (we'll see).
 *   model:    given a recipe and an execution environment, build a simulation.
 *
 * Models will wrap the cell groups etc. and provide sampler implementations
 * to attach to probes etc.
 *
 * Consider thinking of these as concepts, and provide concrete examples?
 * For now, abstract base class while working out the api.
 */

#include <cstddef>
#include <cmath>
#include <random>
#include <vector>
#include <stdexcept>
#include <utility>

#include <cell.hpp>
#include <util/debug.hpp>

namespace nest {
namespace mc {

using cell_id_type = std::size_t;

struct cell_count_info {
    std::size_t num_sources;
    std::size_t num_targets;
    std::size_t num_probes;
};

class invalid_recipe_error: public std::runtime_error {
public:
    invalid_recipe_error(std::string whatstr): std::runtime_error(std::move(whatstr)) {}
};

/* recipe descriptions are cell-oriented: in order that the building
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

// move miniapp's make_cell() into here, but use hashing rng or similar
// to get repeatable recipes
template <typename Rng>
cell make_basic_cell(int compartments_per_segment, int num_synapses, const std::string& syn_type, Rng &);

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

class basic_cell_recipe: public recipe {
public:
    basic_cell_recipe(cell_id_type ncell, basic_recipe_param param, probe_distribution pdist):
        ncell_(ncell), param_(std::move(param)), pdist_(std::move(pdist))
    {
        delay_distribution_param = exp_param{param_.mean_connection_delay_ms
                            - param_.min_connection_delay_ms};
    }

    cell get_cell(cell_id_type i) const override {
        auto gen = std::mt19937(i); // replace this with hashing generator...

        auto cc = get_cell_count_info(i);
        auto cell = make_basic_cell(param_.num_compartments, cc.num_targets,
                        param_.synapse_type, gen);

        EXPECTS(cell.num_segments()==basic_cell_segments);
        EXPECTS(cell.probes().size()==0);
        EXPECTS(cell.synapses().size()==cc.num_targets);
        EXPECTS(cell.detectors().size()==cc.num_sources);

        // add probes
        int n_probe_segs = pdist_.all_segments? basic_cell_segments: 1;
        for (int i = 0; i<n_probe_segs; ++i) {
            if (pdist_.membrane_voltage) {
                cell.add_probe({i, i? 0.5: 0.0}, mc::probeKind::membrane_voltage);
            }
            if (pdist_.membrane_current) {
                cell.add_probe({i, i? 0.5: 0.0}, mc::probeKind::membrane_current);
            }
        }
        EXPECTS(cell.probes().size()==cc.num_probes);
        return cell;
    }

    cell_count_info get_cell_count_info(cell_id_type i) const override {
        cell_count_info cc = {1, std::size_t(param_.num_synapses), 0 };

        // probe this cell?
        if (std::floor(i*pdist_.proportion)!=std::floor((i-1.0)*pdist_.proportion)) {
            std::size_t np = pdist_.membrane_voltage + pdist_.membrane_current;
            if (pdist_.all_segments) {
                np *= basic_cell_segments;
            }

            cc.num_probes = np;
        }

        return cc;
    }

protected:
    template <typename Rng>
    cell_connection draw_connection_params(Rng& rng) const {
        std::exponential_distribution<float> delay_dist(delay_distribution_param);
        float delay = param_.min_connection_delay_ms + delay_dist(rng);
        float weight = param_.syn_weight_per_cell/param_.num_synapses;
        return cell_connection{{0, 0}, {0, 0}, weight, delay};
    }

    cell_id_type ncell_;
    basic_recipe_param param_;
    probe_distribution pdist_;
    static constexpr int basic_cell_segments = 3;

    using exp_param = std::exponential_distribution<float>::param_type;
    exp_param delay_distribution_param;
};

class basic_ring_recipe: public basic_cell_recipe {
public:
    basic_ring_recipe(cell_id_type ncell,
                      basic_recipe_param param,
                      probe_distribution pdist = probe_distribution{}):
        basic_cell_recipe(ncell, std::move(param), std::move(pdist)) {}

    std::vector<cell_connection> connections_on(cell_id_type i) const override {
        std::vector<cell_connection> conns;
        auto gen = std::mt19937(i); // replace this with hashing generator...

        cell_id_type prev = i==0? ncell_-1: i-1;
        for (unsigned t=0; t<param_.num_synapses; ++t) {
            cell_connection cc = draw_connection_params(gen);
            cc.source = {prev, 0};
            cc.dest = {i, t};
            conns.push_back(cc);
        }

        return conns;
    }
};

class basic_rgraph_recipe: public basic_cell_recipe {
public:
    basic_rgraph_recipe(cell_id_type ncell,
                      basic_recipe_param param,
                      std::size_t cell_fan_in,
                      probe_distribution pdist = probe_distribution{}):
        basic_cell_recipe(ncell, std::move(param), std::move(pdist)) {}

    std::vector<cell_connection> connections_on(cell_id_type i) const override {
        std::vector<cell_connection> conns;
        auto conn_param_gen = std::mt19937(i); // replace this with hashing generator...
        auto source_gen = std::mt19937(i*123+457); // ditto

        std::uniform_int_distribution<cell_id_type> source_distribution(0, ncell_-2);

        for (unsigned t=0; t<param_.num_synapses; ++t) {
            auto source = source_distribution(source_gen);
            if (source>=i) ++source;

            cell_connection cc = draw_connection_params(conn_param_gen);
            cc.source = {source, 0};
            cc.dest = {i, t};
            conns.push_back(cc);
        }

        return conns;
    }
};

class basic_kgraph_recipe: public basic_cell_recipe {
public:
    basic_kgraph_recipe(cell_id_type ncell,
                      basic_recipe_param param,
                      probe_distribution pdist = probe_distribution{}):
        basic_cell_recipe(ncell, std::move(param), std::move(pdist))
    {
        if (std::size_t(param.num_synapses) != ncell-1) {
            throw invalid_recipe_error("number of synapses per cell must equal number "
                "of cells minus one in complete graph model");
        }
    }

    std::vector<cell_connection> connections_on(cell_id_type i) const override {
        std::vector<cell_connection> conns;
        auto conn_param_gen = std::mt19937(i); // replace this with hashing generator...

        for (unsigned t=0; t<param_.num_synapses; ++t) {
            cell_id_type source = t>=i? t+1: t;
            EXPECTS(source<ncell_);

            cell_connection cc = draw_connection_params(conn_param_gen);
            cc.source = {source, 0};
            cc.dest = {i, t};
            conns.push_back(cc);
        }

        return conns;
    }
};

} // namespace mc
} // namespace nest
