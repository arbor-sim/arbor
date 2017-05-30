#include <cmath>
#include <random>
#include <vector>
#include <utility>

#include <cell.hpp>
#include <rss_cell.hpp>
#include <morphology.hpp>
#include <util/debug.hpp>

#include "miniapp_recipes.hpp"
#include "morphology_pool.hpp"

namespace nest {
namespace mc {

// TODO: split cell description into separate morphology, stimulus, probes, mechanisms etc.
// description for greater data reuse.

template <typename RNG>
cell make_basic_cell(
    const morphology& morph,
    unsigned compartments_per_segment,
    unsigned num_synapses,
    const std::string& syn_type,
    RNG& rng)
{
    nest::mc::cell cell = make_cell(morph, true);

    for (auto& segment: cell.segments()) {
        if (compartments_per_segment!=0) {
            if (cable_segment* cable = segment->as_cable()) {
                cable->set_compartments(compartments_per_segment);
            }
        }

        if (segment->is_dendrite()) {
            segment->add_mechanism(mc::pas_parameters());
            segment->mechanism("membrane").set("r_L", 100);
        }
    }

    cell.soma()->add_mechanism(mc::hh_parameters());
    cell.add_detector({0,0}, 20);

    auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);

    // Distribute the synapses at random locations the terminal dendrites in a
    // round robin manner.

    morph.assert_valid();
    std::vector<unsigned> terminals;
    for (const auto& section: morph.sections) {
        // Note that morphology section ids should match up exactly with cell
        // segment ids!
        if (section.terminal) {
            terminals.push_back(section.id);
        }
    }

    EXPECTS(!terminals.empty());

    nest::mc::parameter_list syn_default(syn_type);
    for (unsigned i=0; i<num_synapses; ++i) {
        unsigned id = terminals[i%terminals.size()];
        cell.add_synapse({id, distribution(rng)}, syn_default);
    }

    return cell;
}

class basic_cell_recipe: public recipe {
public:
    basic_cell_recipe(cell_gid_type ncell, basic_recipe_param param, probe_distribution pdist):
        ncell_(ncell), param_(std::move(param)), pdist_(std::move(pdist))
    {
        EXPECTS(param_.morphologies.size()>0);
        delay_distribution_param_ = exp_param{param_.mean_connection_delay_ms
                            - param_.min_connection_delay_ms};
    }

    cell_size_type num_cells() const override { return ncell_; }

    util::unique_any get_cell_description(cell_gid_type i) const override {
        // The last 'cell' is always a regular spiking neuron
        // That spikes only once at t=0
        if (i == ncell_-1) {
            return util::unique_any(std::move(
                rss_cell::rss_cell_descr(0.0, 0.1, 0.1) ));
        }

        auto gen = std::mt19937(i); // TODO: replace this with hashing generator...

        auto cc = get_cell_count_info(i);
        const auto& morph = get_morphology(i);
        unsigned cell_segments = morph.components();

        auto cell = make_basic_cell(morph, param_.num_compartments, cc.num_targets,
                        param_.synapse_type, gen);

        EXPECTS(cell.num_segments()==cell_segments);
        EXPECTS(cell.probes().size()==0);
        EXPECTS(cell.synapses().size()==cc.num_targets);
        EXPECTS(cell.detectors().size()==cc.num_sources);

        // add probes
        if (cc.num_probes) {
            unsigned n_probe_segs = pdist_.all_segments? cell_segments: 1u;
            for (unsigned i = 0; i<n_probe_segs; ++i) {
                if (pdist_.membrane_voltage) {
                    cell.add_probe({{i, i? 0.5: 0.0}, mc::probeKind::membrane_voltage});
                }
                if (pdist_.membrane_current) {
                    cell.add_probe({{i, i? 0.5: 0.0}, mc::probeKind::membrane_current});
                }
            }
        }
        EXPECTS(cell.probes().size()==cc.num_probes);

        return util::unique_any(std::move(cell));
    }

    cell_kind get_cell_kind(cell_gid_type i ) const override {
        // First cell is currently always a regular frequency neuron
        if (i == ncell_-1) {
            return cell_kind::regular_spike_source;
        }
        return cell_kind::cable1d_neuron;
    }

    cell_count_info get_cell_count_info(cell_gid_type i) const override {
        cell_count_info cc = {1, param_.num_synapses, 0 };
        unsigned cell_segments = get_morphology(i).components();

        // probe this cell?
        if (std::floor(i*pdist_.proportion)!=std::floor((i-1.0)*pdist_.proportion)) {
            std::size_t np = pdist_.membrane_voltage + pdist_.membrane_current;
            if (pdist_.all_segments) {
                np *= cell_segments;
            }

            cc.num_probes = np;
        }

        return cc;
    }

protected:
    template <typename RNG>
    cell_connection draw_connection_params(RNG& rng) const {
        std::exponential_distribution<float> delay_dist(delay_distribution_param_);
        float delay = param_.min_connection_delay_ms + delay_dist(rng);
        float weight = param_.syn_weight_per_cell/param_.num_synapses;
        return cell_connection{{0, 0}, {0, 0}, weight, delay};
    }

    cell_gid_type ncell_;
    basic_recipe_param param_;
    probe_distribution pdist_;

    using exp_param = std::exponential_distribution<float>::param_type;
    exp_param delay_distribution_param_;

    const morphology& get_morphology(cell_gid_type gid) const {
        // Allocate to gids sequentially?
        if (param_.morphology_round_robin) {
            return param_.morphologies[gid%param_.morphologies.size()];
        }

        // Morphologies are otherwise selected deterministically pseudo-randomly from pool.
        std::uniform_int_distribution<unsigned> morph_select_dist_(0, param_.morphologies.size()-1);

        // TODO: definitely replace this with a random hash!
        auto gen = std::mt19937(gid+0xbad0cafe);
        return param_.morphologies[morph_select_dist_(gen)];
    }
};

class basic_ring_recipe: public basic_cell_recipe {
public:
    basic_ring_recipe(cell_gid_type ncell,
                      basic_recipe_param param,
                      probe_distribution pdist = probe_distribution{}):
        basic_cell_recipe(ncell, std::move(param), std::move(pdist)) {}

    std::vector<cell_connection> connections_on(cell_gid_type i) const override {
        std::vector<cell_connection> conns;

        // The frequency spiking does not have inputs
        if (i == ncell_-1) {
            return conns;
        }

        auto gen = std::mt19937(i); // TODO: replace this with hashing generator...

        cell_gid_type prev = i==0? ncell_-2: i-1;
        for (unsigned t=0; t<param_.num_synapses; ++t) {
            cell_connection cc = draw_connection_params(gen);
            cc.source = {prev, 0};
            cc.dest = {i, t};
            conns.push_back(cc);

            // Each 20th neuron generates a artificial spike. We now generate these
            // in a separate artificial spiking neuron. We need at add a mirror of
            // each synapse from these neurons on gid=0 to
            // reproduce this results
            if (prev % 20 == 0) {
                cc.source = { ncell_-1, 0 }; // also add connection from reg spiker!
                conns.push_back(cc);
            }
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

        // The frequency spiking does not have inputs
        if (i == ncell_-1) {
            return conns;
        }
        auto conn_param_gen = std::mt19937(i); // TODO: replace this with hashing generator...
        auto source_gen = std::mt19937(i*123+457); // ditto

        // TODO: In the original implementation this is:
        // std::uniform_int_distribution<cell_gid_type> source_distribution(0, ncell_-2);
        // Why, how, when
        std::uniform_int_distribution<cell_gid_type> source_distribution(0, ncell_-3);

        for (unsigned t=0; t<param_.num_synapses; ++t) {
            auto source = source_distribution(source_gen);
            if (source>=i) ++source;

            cell_connection cc = draw_connection_params(conn_param_gen);
            cc.source = {source, 0};
            cc.dest = {i, t};
            conns.push_back(cc);

            // Each 20th neuron generates a artificial spike. We now generate these
            // in a separate artificial spiking neuron. We need at add a mirror of
            // each synapse from these neurons on gid=0 to
            // reproduce this results
            if ((source % 20) == 0) {
                cc.source = { ncell_ - 1, 0 }; // also add connection from reg spiker!
                conns.push_back(cc);
            }
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
        if (std::size_t(param.num_synapses) != (ncell-2)) {
            throw invalid_recipe_error("number of synapses per cell must equal number "
                "of cells minus one in complete graph model");
        }
    }

    std::vector<cell_connection> connections_on(cell_gid_type i) const override {
        std::vector<cell_connection> conns;
        // The frequency spiking does not have inputs
        if (i == ncell_-1) {
            return conns;
        }
        auto conn_param_gen = std::mt19937(i); // TODO: replace this with hashing generator...

        for (unsigned t=0; t<param_.num_synapses; ++t) {
            cell_gid_type source = t>=i? t+1: t;

            EXPECTS(source<(ncell_ - 1));

            cell_connection cc = draw_connection_params(conn_param_gen);
            cc.source = {source, 0};
            cc.dest = {i, t};
            conns.push_back(cc);

            // Each 20th neuron generates a artificial spike. We now generate these
            // in a separate artificial spiking neuron. We need at add a mirror of
            // each synapse from these neurons on gid=0 to
            // reproduce this results
            if ((source % 20) == 0) {
                cc.source = { ncell_ - 1, 0 }; // also add connection from reg spiker!
                conns.push_back(cc);
            }
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
