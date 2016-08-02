#include <cmath>
#include <random>
#include <vector>
#include <utility>

#include <cell.hpp>
#include <util/debug.hpp>

#include "miniapp_recipes.hpp"

namespace nest {
namespace mc {

// TODO: split cell description into separate morphology, stimulus, probes, mechanisms etc.
// description for greater data reuse.

template <typename RNG>
mc::cell make_basic_cell(unsigned compartments_per_segment, unsigned num_synapses, const std::string& syn_type, RNG& rng) {
    nest::mc::cell cell;

    // Soma with diameter 12.6157 um and HH channel
    auto soma = cell.add_soma(12.6157/2.0);
    soma->add_mechanism(mc::hh_parameters());

    // add dendrite of length 200 um and diameter 1 um with passive channel
    std::vector<mc::cable_segment*> dendrites;
    dendrites.push_back(cell.add_cable(0, mc::segmentKind::dendrite, 0.5, 0.5, 200));
    dendrites.push_back(cell.add_cable(1, mc::segmentKind::dendrite, 0.5, 0.25,100));
    dendrites.push_back(cell.add_cable(1, mc::segmentKind::dendrite, 0.5, 0.25,100));

    for (auto d : dendrites) {
        d->add_mechanism(mc::pas_parameters());
        d->set_compartments(compartments_per_segment);
        d->mechanism("membrane").set("r_L", 100);
    }

    cell.add_detector({0,0}, 20);

    auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);
    // distribute the synapses at random locations the terminal dendrites in a
    // round robin manner
    nest::mc::parameter_list syn_default(syn_type);
    for (unsigned i=0; i<num_synapses; ++i) {
        cell.add_synapse({2+(i%2), distribution(rng)}, syn_default);
    }

    return cell;
}

class basic_cell_recipe: public recipe {
public:
    basic_cell_recipe(cell_gid_type ncell, basic_recipe_param param, probe_distribution pdist):
        ncell_(ncell), param_(std::move(param)), pdist_(std::move(pdist))
    {
        delay_distribution_param = exp_param{param_.mean_connection_delay_ms
                            - param_.min_connection_delay_ms};
    }

    cell_size_type num_cells() const override { return ncell_; }

    cell get_cell(cell_gid_type i) const override {
        auto gen = std::mt19937(i); // replace this with hashing generator...

        auto cc = get_cell_count_info(i);
        auto cell = make_basic_cell(param_.num_compartments, cc.num_targets,
                        param_.synapse_type, gen);

        EXPECTS(cell.num_segments()==basic_cell_segments);
        EXPECTS(cell.probes().size()==0);
        EXPECTS(cell.synapses().size()==cc.num_targets);
        EXPECTS(cell.detectors().size()==cc.num_sources);

        // add probes
        unsigned n_probe_segs = pdist_.all_segments? basic_cell_segments: 1u;
        for (unsigned i = 0; i<n_probe_segs; ++i) {
            if (pdist_.membrane_voltage) {
                cell.add_probe({{i, i? 0.5: 0.0}, mc::probeKind::membrane_voltage});
            }
            if (pdist_.membrane_current) {
                cell.add_probe({{i, i? 0.5: 0.0}, mc::probeKind::membrane_current});
            }
        }
        EXPECTS(cell.probes().size()==cc.num_probes);
        return cell;
    }

    cell_count_info get_cell_count_info(cell_gid_type i) const override {
        cell_count_info cc = {1, param_.num_synapses, 0 };

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
    template <typename RNG>
    cell_connection draw_connection_params(RNG& rng) const {
        std::exponential_distribution<float> delay_dist(delay_distribution_param);
        float delay = param_.min_connection_delay_ms + delay_dist(rng);
        float weight = param_.syn_weight_per_cell/param_.num_synapses;
        return cell_connection{{0, 0}, {0, 0}, weight, delay};
    }

    cell_gid_type ncell_;
    basic_recipe_param param_;
    probe_distribution pdist_;
    static constexpr int basic_cell_segments = 4;

    using exp_param = std::exponential_distribution<float>::param_type;
    exp_param delay_distribution_param;
};

class basic_ring_recipe: public basic_cell_recipe {
public:
    basic_ring_recipe(cell_gid_type ncell,
                      basic_recipe_param param,
                      probe_distribution pdist = probe_distribution{}):
        basic_cell_recipe(ncell, std::move(param), std::move(pdist)) {}

    std::vector<cell_connection> connections_on(cell_gid_type i) const override {
        std::vector<cell_connection> conns;
        auto gen = std::mt19937(i); // replace this with hashing generator...

        cell_gid_type prev = i==0? ncell_-1: i-1;
        for (unsigned t=0; t<param_.num_synapses; ++t) {
            cell_connection cc = draw_connection_params(gen);
            cc.source = {prev, 0};
            cc.dest = {i, t};
            conns.push_back(cc);
        }

        return conns;
    }
};

std::unique_ptr<recipe> make_basic_ring_recipe(
        cell_gid_type ncell,
        basic_recipe_param param,
        probe_distribution pdist)
{
    return std::unique_ptr<recipe>(new basic_ring_recipe(ncell, param, pdist));
}


class basic_rgraph_recipe: public basic_cell_recipe {
public:
    basic_rgraph_recipe(cell_gid_type ncell,
                      basic_recipe_param param,
                      probe_distribution pdist = probe_distribution{}):
        basic_cell_recipe(ncell, std::move(param), std::move(pdist)) {}

    std::vector<cell_connection> connections_on(cell_gid_type i) const override {
        std::vector<cell_connection> conns;
        auto conn_param_gen = std::mt19937(i); // replace this with hashing generator...
        auto source_gen = std::mt19937(i*123+457); // ditto

        std::uniform_int_distribution<cell_gid_type> source_distribution(0, ncell_-2);

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

std::unique_ptr<recipe> make_basic_rgraph_recipe(
        cell_gid_type ncell,
        basic_recipe_param param,
        probe_distribution pdist)
{
    return std::unique_ptr<recipe>(new basic_rgraph_recipe(ncell, param, pdist));
}

class basic_kgraph_recipe: public basic_cell_recipe {
public:
    basic_kgraph_recipe(cell_gid_type ncell,
                      basic_recipe_param param,
                      probe_distribution pdist = probe_distribution{}):
        basic_cell_recipe(ncell, std::move(param), std::move(pdist))
    {
        if (std::size_t(param.num_synapses) != ncell-1) {
            throw invalid_recipe_error("number of synapses per cell must equal number "
                "of cells minus one in complete graph model");
        }
    }

    std::vector<cell_connection> connections_on(cell_gid_type i) const override {
        std::vector<cell_connection> conns;
        auto conn_param_gen = std::mt19937(i); // replace this with hashing generator...

        for (unsigned t=0; t<param_.num_synapses; ++t) {
            cell_gid_type source = t>=i? t+1: t;
            EXPECTS(source<ncell_);

            cell_connection cc = draw_connection_params(conn_param_gen);
            cc.source = {source, 0};
            cc.dest = {i, t};
            conns.push_back(cc);
        }

        return conns;
    }
};

std::unique_ptr<recipe> make_basic_kgraph_recipe(
        cell_gid_type ncell,
        basic_recipe_param param,
        probe_distribution pdist)
{
    return std::unique_ptr<recipe>(new basic_kgraph_recipe(ncell, param, pdist));
}

} // namespace mc
} // namespace nest
