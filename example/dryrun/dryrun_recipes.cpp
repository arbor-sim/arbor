#include <cmath>
#include <random>
#include <vector>
#include <utility>

#include <arbor/assert.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/morphology.hpp>
#include <arbor/schedule.hpp>
#include <arbor/spike_source_cell.hpp>


#include "io.hpp"
#include "dryrun_recipes.hpp"
#include "morphology_desc.hpp"

namespace arb {

template <typename RNG>
mc_cell make_basic_cell_tiled(
    const morphology& morph,
    unsigned compartments_per_segment,
    unsigned num_synapses,
    const std::string& syn_type,
    RNG& rng)
{
    mc_cell cell = make_mc_cell(morph, true);

    for (auto& segment: cell.segments()) {
        if (compartments_per_segment!=0) {
            if (cable_segment* cable = segment->as_cable()) {
                cable->set_compartments(compartments_per_segment);
            }
        }

        if (segment->is_dendrite()) {
            segment->add_mechanism("pas");
            segment->rL = 100;
        }
    }

    cell.soma()->add_mechanism("hh");
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

    arb_assert(!terminals.empty());

    arb::mechanism_desc syn_default(syn_type);
    for (unsigned i=0; i<num_synapses; ++i) {
        unsigned id = terminals[i%terminals.size()];
        cell.add_synapse({id, distribution(rng)}, syn_default);
    }

    return cell;
}

class basic_tiled_recipe: public tiled_recipe {
public:
    basic_tiled_recipe(cell_gid_type ncell, cell_gid_type ntiles, basic_recipe_param param, probe_distribution pdist):
        ncell_(ncell), ntiles_(ntiles), param_(std::move(param)), pdist_(std::move(pdist))
    {
        delay_distribution_param_ = exp_param{param_.mean_connection_delay_ms
                            - param_.min_connection_delay_ms};
    }

    cell_size_type num_tiles() const override {
        return ntiles_;
    }

    cell_size_type num_cells() const override {
        return ncell_ + 1;  // We automatically add a fake cell to each recipe!
    }

    util::unique_any get_cell_description(cell_gid_type i) const override {
        // The last 'cell' is a spike source cell, producing one spike at t = 0.
        if (i == ncell_) {
            return util::unique_any(spike_source_cell{explicit_schedule({0.})});
        }

        auto gen = std::mt19937(i); // TODO: replace this with hashing generator...

        unsigned cell_segments = param_.morph.components();

        auto cell = make_basic_cell_tiled(param_.morph, param_.num_compartments, param_.num_synapses,
                        param_.synapse_type, gen);

        arb_assert(cell.num_segments()==cell_segments);
        arb_assert(cell.synapses().size()==num_targets(i));
        arb_assert(cell.detectors().size()==num_sources(i));

        return util::unique_any(std::move(cell));
    }

    probe_info get_probe(cell_member_type probe_id) const override {
        if (probe_id.index>=num_probes(probe_id.gid)) {
            throw arb::bad_probe_id(probe_id);
        }

        // if we have both voltage and current probes, then order them
        // voltage compartment 0, current compartment 0, voltage compartment 1, ...

        cell_probe_address::probe_kind kind;

        int stride = pdist_.membrane_voltage+pdist_.membrane_current;

        if (stride==1) {
            // Just one kind of probe.
            kind = pdist_.membrane_voltage?
                cell_probe_address::membrane_voltage: cell_probe_address::membrane_current;
        }
        else {
            arb_assert(stride==2);
            // Both kinds available.
            kind = (probe_id.index%stride==0)?
                cell_probe_address::membrane_voltage: cell_probe_address::membrane_current;
        }

        cell_lid_type compartment = probe_id.index/stride;
        segment_location loc{compartment, compartment? 0.5: 0.0};

        // Use probe kind as the token to be passed to a sampler.
        return {probe_id, (int)kind, cell_probe_address{loc, kind}};
    }

    cell_kind get_cell_kind(cell_gid_type i) const override {
        // The last 'cell' is a regular spike source with one spike at t=0
        if (i == ncell_) {
            return cell_kind::spike_source;
        }
        return cell_kind::cable1d_neuron;
    }

    cell_size_type num_sources(cell_gid_type i) const override {
        return 1;
    }

    cell_size_type num_targets(cell_gid_type i) const override {
        return param_.num_synapses;
    }

    cell_size_type num_probes(cell_gid_type i) const override {
        bool has_probe = (std::floor(i*pdist_.proportion)!=std::floor((i-1.0)*pdist_.proportion));

        if (!has_probe) {
            return 0;
        }
        else {
            cell_size_type np = pdist_.all_segments? param_.morph.components(): 1;
            return np*(pdist_.membrane_voltage+pdist_.membrane_current);
        }
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
    cell_gid_type ntiles_;
    basic_recipe_param param_;
    probe_distribution pdist_;

    using exp_param = std::exponential_distribution<float>::param_type;
    exp_param delay_distribution_param_;
};

class basic_rgraph_tiled_recipe: public basic_tiled_recipe {
public:
    basic_rgraph_tiled_recipe(cell_gid_type ncell,
                      cell_gid_type ntiles,
                      basic_recipe_param param,
                      probe_distribution pdist = probe_distribution{}):
        basic_tiled_recipe(ncell, ntiles, std::move(param), std::move(pdist))
    {
        // Cells are not allowed to connect to themselves; hence there must be least two cells
        // to build a connected network.
        if (ncell<2) {
            throw std::runtime_error("A randomly connected network must have at least 2 cells.");
        }
    }

    std::vector<cell_connection> connections_on(cell_gid_type i) const override {
        std::vector<cell_connection> conns;

        // The regular spike cell does not have inputs
        if (i == ncell_) {
            return conns;
        }
        auto conn_param_gen = std::mt19937(i); // TODO: replace this with hashing generator...
        auto rank_gen = std::mt19937(i*153*687); // ditto
        auto source_gen = std::mt19937(i*123+457); // ditto

        std::uniform_int_distribution<cell_gid_type> rank_distribution(0, ntiles_-1);

        for (unsigned t=0; t<param_.num_synapses; ++t) {
            auto rank = rank_distribution(rank_gen);
            arb::cell_gid_type source;
            if(rank == 0) {
                std::uniform_int_distribution<cell_gid_type> source_distribution(0, ncell_-2);
                source = source_distribution(source_gen);
                if (source>=i) ++source;
            }
            else {
                std::uniform_int_distribution<cell_gid_type> source_distribution((ncell_ + 1)*rank, (ncell_ + 1)*(rank+ 1) - 2);
                source = source_distribution(source_gen);
            }
            cell_connection cc = draw_connection_params(conn_param_gen);
            cc.source = {source, 0};
            cc.dest = {i, t};
            conns.push_back(cc);

            // The regular spike source spikes at t=0, with these connections it looks like
            // (source % 20) == 0 spikes at that moment.
            if (source % 20 == 0) {
                cc.source = {ncell_, 0};
                conns.push_back(cc);
            }
        }

        return conns;
    }
};

std::unique_ptr<recipe> make_symmetric_recipe(basic_rgraph_tiled_recipe rec) {
    return std::unique_ptr<recipe>(new symmetric_recipe<basic_rgraph_tiled_recipe>(rec));
}

std::unique_ptr<recipe> make_basic_rgraph_symmetric_recipe(
        cell_gid_type ncell,
        cell_gid_type ntiles,
        basic_recipe_param param,
        probe_distribution pdist)
{
    return make_symmetric_recipe(basic_rgraph_tiled_recipe(ncell, ntiles, param, pdist));
}

} // namespace arb
