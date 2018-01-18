/*
 * A miniapp that demonstrates how to use event generators.
 *
 * The miniapp builds a simple model of a single cell, with one compartment
 * corresponding to the soma. The soma has a single synapse, to which two
 * event generators, one inhibitory, and one excitatory, are attached.
 */

#include <fstream>
#include <iomanip>
#include <iostream>

#include <json/json.hpp>

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

// Writes voltage trace as a json file.
void write_trace_json(const arb::trace_data<double>& trace);

class generator_recipe: public arb::recipe {
public:
    // There is just the one cell in the model
    cell_size_type num_cells() const override {
        return 1;
    }

    // Create cell with just a single compartment for the soma:
    //    soma diameter: 18.8 µm
    //    mechanisms: pas [default params]
    //    bulk resistivitiy: 100 Ω·cm [default]
    //    capacitance: 0.01 F/m² [default]
    //    synapses: 1 * expsyn
    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        arb::cell c;

        c.add_soma(18.8/2.0); // convert 18.8 μm diameter to radius
        c.soma()->add_mechanism("pas");

        // Add one synapse at the soma.
        // This synapse will be the target for all events, from both
        // event_generators.
        auto syn_spec = arb::mechanism_spec("expsyn");
        c.add_synapse({0, 0.5}, syn_spec);

        return std::move(c);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        EXPECTS(gid==0); // There is only one cell in the model
        return cell_kind::cable1d_neuron;
    }

    // The cell has one target synapse, which receives both inhibitory and exchitatory inputs.
    cell_size_type num_targets(cell_gid_type gid) const override {
        EXPECTS(gid==0); // There is only one cell in the model
        return 1;
    }

    // Return two generators attached to the one cell.
    std::vector<arb::event_generator_ptr> event_generators(cell_gid_type gid) const override {
        EXPECTS(gid==0); // There is only one cell in the model

        using RNG = std::mt19937_64;
        using pgen = arb::poisson_generator<RNG>;

        auto hz_to_freq = [](double hz) { return hz*1e-3; };
        time_type t0 = 0;

        // Define frequencies and weights for the excitatory and inhibitory generators.
        double lambda_e =  hz_to_freq(500);
        double lambda_i =  hz_to_freq(20);
        double w_e =  0.001;
        double w_i = -0.005;

        // Make two event generators.
        std::vector<arb::event_generator_ptr> gens;

        // Add excitatory generator
        gens.push_back(
            arb::make_event_generator<pgen>(
                cell_member_type{0,0}, // Target synapse (gid, local_id).
                w_e,                   // Weight of events to deliver
                RNG(29562872),         // Random number generator to use
                t0,                    // Events start being delivered from this time
                lambda_e));            // Expected frequency (events per ms)

        // Add inhibitory generator
        gens.push_back(
            arb::make_event_generator<pgen>(
                cell_member_type{0,0}, w_i, RNG(86543891), t0, lambda_i));

        return gens;
    }

    // There is one probe (for measuring voltage at the soma) on the cell
    cell_size_type num_probes(cell_gid_type gid)  const override {
        EXPECTS(gid==0); // There is only one cell in the model
        return 1;
    }

    arb::probe_info get_probe(cell_member_type id) const override {
        EXPECTS(id.gid==0);     // There is one cell,
        EXPECTS(id.index==0);   // with one probe.

        // Get the appropriate kind for measuring voltage
        cell_probe_address::probe_kind kind = cell_probe_address::membrane_voltage;
        // Measure at the soma
        arb::segment_location loc(0, 0.0);

        return arb::probe_info{id, kind, cell_probe_address{loc, kind}};
    }
};

int main() {
    // Create an instance of our recipe.
    generator_recipe recipe;

    // Make the domain decomposition for the model
    auto node = arb::hw::get_node_info();
    auto decomp = arb::partition_load_balance(recipe, node);

    // Construct the model.
    arb::model model(recipe, decomp);

    // Set up the probe that will measure voltage in the cell.

    // The id of the only probe on the cell: the cell_member type points to (cell 0, probe 0)
    auto probe_id = cell_member_type{0, 0};
    // The schedule for sampling is 10 samples every 1 ms.
    auto sched = arb::regular_schedule(0.1);
    // This is where the voltage samples will be stored as (time, value) pairs
    arb::trace_data<double> voltage;
    // Now attach the sampler at probe_id, with sampling schedule sched, writing to voltage
    model.add_sampler(arb::one_probe(probe_id), sched, arb::make_simple_sampler(voltage));

    // Run the model for 1 s (1000 ms), with time steps of 0.01 ms.
    model.run(50, 0.01);

    // Write the samples to a json file.
    write_trace_json(voltage);
}

void write_trace_json(const arb::trace_data<double>& trace) {
    std::string path = "./voltages.json";

    nlohmann::json json;
    json["name"] = "event_gen_demo";
    json["units"] = "mV";
    json["cell"] = "0.0";
    json["probe"] = "0";

    auto& jt = json["data"]["time"];
    auto& jy = json["data"]["voltage"];

    for (const auto& sample: trace) {
        jt.push_back(sample.t);
        jy.push_back(sample.v);
    }

    std::ofstream file(path);
    file << std::setw(1) << json << "\n";
}

