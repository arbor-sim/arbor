/*
 * A miniapp that demonstrates how to use event generators.
 *
 * The miniapp builds a simple model of a single cell, with one compartment
 * corresponding to the soma. The soma has a single synapse, to which two 
 * event generators, one inhibitory, and one excitatory, are attached.
 */

#include <cell.hpp>
#include <common_types.hpp>
#include <event_generator.hpp>
#include <hardware/node_info.hpp>
#include <load_balance.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <simple_sampler.hpp>

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

class generator_recipe: public arb::recipe {
public:
    cell_size_type num_cells() const override {
        return 1;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        // Create cell with just a single compartment for the soma:
        //    soma diameter: 18.8 µm
        //    mechanisms: pas (default params)
        //    bulk resistivitiy: 100 Ω·cm [default]
        //    capacitance: 0.01 F/m² [default]
        //    synapses: 2 * expsyn
        arb::cell c;

        c.add_soma(18.8/2.0);
        c.soma()->add_mechanism("pas");

        // Add one synapse at the soma.
        // This synapse will be the target for all events, from both
        // event_generators.
        auto syn_spec = arb::mechanism_spec("expsyn");
        c.add_synapse({0, 0.5}, syn_spec);

        return c;
    }

    cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable1d_neuron;
    }
    cell_size_type num_sources(cell_gid_type) const override {
        return 0;
    }

    // The cell has one target synapse, which receives both inhibitory and exchitatory inputs.
    cell_size_type num_targets(cell_gid_type) const override {
        return 1;
    }

    // Return two generators attached to the one cell 
    std::vector<arb::event_generator_ptr> event_generators(cell_gid_type gid) const override {
        using RNG = std::mt19937_64;
        using pgen = arb::poisson_generator<RNG>;

        auto hz_to_freq = [](double hz) { return hz*1e-3; };
        time_type t0 = 0;
        double e_weight =  10;
        double i_weight = -10;

        // Make two event generators.
        std::vector<arb::event_generator_ptr> gens;
        gens.push_back(
            arb::make_event_generator<pgen>(
                cell_member_type{0,0}, // target synapse (gid, local_id)
                e_weight,              // weight of events to deliver
                RNG(29562872),         // random number generator to use
                t0,                    // events start being delivered from this time
                hz_to_freq(50)));      // 50 Hz average firing rate
        gens.push_back(
            arb::make_event_generator<pgen>(
                cell_member_type{0,0}, i_weight, RNG(86543891), t0, hz_to_freq(50)));

        return gens;
    }

    // There is one probe (for measuring voltage at the soma) on the cell
    cell_size_type num_probes(cell_gid_type gid)  const override {
        EXPECTS(gid==0); // There is only one cell in the model
        return 1;
    }

    arb::probe_info get_probe(cell_member_type id) const override {
        EXPECTS(id.gid==0);     // There is one cell...
        EXPECTS(id.index==0);   // with one probe, in the model.

        // Get the appropriate kind for measuring voltage
        cell_probe_address::probe_kind kind = cell_probe_address::membrane_voltage;
        // Measure at the soma
        arb::segment_location loc(0, 0.0);

        return arb::probe_info{id, kind, cell_probe_address{loc, kind}};
    }
};

int main() {
    generator_recipe recipe;

    auto node = arb::hw::get_node_info();
    auto decomp = arb::partition_load_balance(recipe, node);
    arb::model model(recipe, decomp);

    auto probe_id = cell_member_type{0, 0};
    auto probe = recipe.get_probe(probe_id);

    arb::trace_data<double> voltage;

    // sample every ms
    auto sched = arb::regular_schedule(1);
    model.add_sampler(arb::one_probe(probe_id), sched, arb::make_simple_sampler(voltage));

    model.run(100, 0.01);

    std::cout << "at the end we have " << voltage.size() << " samples\n";
    for (auto v: voltage) {
        std::cout << v.t << ": " << v.v << "\n";
    }
}
