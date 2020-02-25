/*
 * A miniapp that demonstrates how to make a model with gap junctions
 *
 */

#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>
#include <arbor/version.hpp>

#include <arborenv/concurrency.hpp>
#include <arborenv/gpu_env.hpp>

#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>
#include <sup/json_params.hpp>

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <arborenv/with_mpi.hpp>
#endif

struct gap_params {
    std::string name = "default";
    unsigned n_cables = 3;
    unsigned n_cells_per_cable = 5;
    double stim_duration = 30;
    double event_min_delay = 10;
    double event_weight = 0.05;
    double sim_duration = 100;
    bool print_all = true;
};

gap_params read_options(int argc, char** argv);

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

// Writes voltage trace as a json file.
void write_trace_json(const std::vector<arb::trace_data<double>>& trace, unsigned rank);

// Generate a cell.
arb::cable_cell gj_cell(cell_gid_type gid, unsigned ncells, double stim_duration);

class gj_recipe: public arb::recipe {
public:
    gj_recipe(gap_params params): params_(params) {}

    cell_size_type num_cells() const override {
        return params_.n_cells_per_cable * params_.n_cables;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        return gj_cell(gid, params_.n_cells_per_cable, params_.stim_duration);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable;
    }

    // Each cell has one spike detector (at the soma).
    cell_size_type num_sources(cell_gid_type gid) const override {
        return 1;
    }

    // The cell has one target synapse, which will be connected to a cell in another cable.
    cell_size_type num_targets(cell_gid_type gid) const override {
        return 1;
    }

    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        if(gid % params_.n_cells_per_cable || (int)gid - 1 < 0) {
            return{};
        }
        return {arb::cell_connection({gid - 1, 0}, {gid, 0}, params_.event_weight, params_.event_min_delay)};
    }

    // There is one probe (for measuring voltage at the soma) on the cell.
    cell_size_type num_probes(cell_gid_type gid)  const override {
        return 1;
    }

    arb::probe_info get_probe(cell_member_type id) const override {
        // Get the appropriate kind for measuring voltage.
        cell_probe_address::probe_kind kind = cell_probe_address::membrane_voltage;
        // Measure at the soma.
        arb::mlocation loc{0, 1.};

        return arb::probe_info{id, kind, cell_probe_address{loc, kind}};
    }

    arb::util::any get_global_properties(cell_kind k) const override {
        arb::cable_cell_global_properties a;
        a.default_parameters = arb::neuron_parameter_defaults;
        a.default_parameters.temperature_K = 308.15;
        return a;
    }

    std::vector<arb::gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override{
        std::vector<arb::gap_junction_connection> conns;

        int cable_begin = (gid/params_.n_cells_per_cable) * params_.n_cells_per_cable;
        int cable_end = cable_begin + params_.n_cells_per_cable;

        int next_cell = gid + 1;
        int prev_cell = gid - 1;

        // Soma is connected to the prev cell's dendrite
        // Dendrite is connected to the next cell's soma
        // Gap junction conductance in μS

        if (next_cell < cable_end) {
            conns.push_back(arb::gap_junction_connection({(cell_gid_type)next_cell, 0}, {gid, 1}, 0.015));
        }
        if (prev_cell >= cable_begin) {
            conns.push_back(arb::gap_junction_connection({(cell_gid_type)prev_cell, 1}, {gid, 0}, 0.015));
        }

        return conns;
    }

private:
    gap_params params_;
};

int main(int argc, char** argv) {
    try {
        bool root = true;

        arb::proc_allocation resources;
        if (auto nt = arbenv::get_env_num_threads()) {
            resources.num_threads = nt;
        }
        else {
            resources.num_threads = arbenv::thread_concurrency();
        }

#ifdef ARB_MPI_ENABLED
        arbenv::with_mpi guard(argc, argv, false);
        resources.gpu_id = arbenv::find_private_gpu(MPI_COMM_WORLD);
        auto context = arb::make_context(resources, MPI_COMM_WORLD);
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            root = rank==0;
        }
#else
        resources.gpu_id = arbenv::default_gpu();
        auto context = arb::make_context(resources);
#endif

#ifdef ARB_PROFILE_ENABLED
        arb::profile::profiler_initialize(context);
#endif

        std::cout << sup::mask_stream(root);

        // Print a banner with information about hardware configuration
        std::cout << "gpu:      " << (has_gpu(context)? "yes": "no") << "\n";
        std::cout << "threads:  " << num_threads(context) << "\n";
        std::cout << "mpi:      " << (has_mpi(context)? "yes": "no") << "\n";
        std::cout << "ranks:    " << num_ranks(context) << "\n" << std::endl;

        auto params = read_options(argc, argv);

        arb::profile::meter_manager meters;
        meters.start(context);

        // Create an instance of our recipe.
        gj_recipe recipe(params);

        auto decomp = arb::partition_load_balance(recipe, context);

        // Construct the model.
        arb::simulation sim(recipe, decomp, context);

        // Set up the probe that will measure voltage in the cell.

        auto sched = arb::regular_schedule(0.025);
        // This is where the voltage samples will be stored as (time, value) pairs
        std::vector<arb::trace_data<double>> voltage(decomp.num_local_cells);

        // Now attach the sampler at probe_id, with sampling schedule sched, writing to voltage
        unsigned j=0;
        for (auto g : decomp.groups) {
            for (auto i : g.gids) {
                auto t = recipe.get_probe({i, 0});
                sim.add_sampler(arb::one_probe(t.id), sched, arb::make_simple_sampler(voltage[j++]));
            }
        }

        // Set up recording of spikes to a vector on the root process.
        std::vector<arb::spike> recorded_spikes;
        if (root) {
            sim.set_global_spike_callback(
                [&recorded_spikes](const std::vector<arb::spike>& spikes) {
                    recorded_spikes.insert(recorded_spikes.end(), spikes.begin(), spikes.end());
                });
        }

        meters.checkpoint("model-init", context);

        std::cout << "running simulation" << std::endl;
        // Run the simulation for 100 ms, with time steps of 0.025 ms.
        sim.run(params.sim_duration, 0.025);

        meters.checkpoint("model-run", context);

        auto ns = sim.num_spikes();

        // Write spikes to file
        if (root) {
            std::cout << "\n" << ns << " spikes generated at rate of "
                      << params.sim_duration/ns << " ms between spikes\n";
            std::ofstream fid("spikes.gdf");
            if (!fid.good()) {
                std::cerr << "Warning: unable to open file spikes.gdf for spike output\n";
            }
            else {
                char linebuf[45];
                for (auto spike: recorded_spikes) {
                    auto n = std::snprintf(
                        linebuf, sizeof(linebuf), "%u %.4f\n",
                        unsigned{spike.source.gid}, float(spike.time));
                    fid.write(linebuf, n);
                }
            }
        }

        // Write the samples to a json file.
        if (params.print_all) {
            write_trace_json(voltage, arb::rank(context));
        }

        auto report = arb::profile::make_meter_report(meters, context);
        std::cout << report;
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in gap junction miniapp:\n" << e.what() << "\n";
        return 1;
    }

    return 0;
}

void write_trace_json(const std::vector<arb::trace_data<double>>& trace, unsigned rank) {
    for (unsigned i = 0; i < trace.size(); i++) {
        std::string path = "./voltages_" + std::to_string(rank) +
                           "_" + std::to_string(i) + ".json";

        nlohmann::json json;
        json["name"] = "gj demo: cell " + std::to_string(i);
        json["units"] = "mV";
        json["cell"] = std::to_string(i);
        json["group"] = std::to_string(rank);
        json["probe"] = "0";

        auto &jt = json["data"]["time"];
        auto &jy = json["data"]["voltage"];

        for (const auto &sample: trace[i]) {
            jt.push_back(sample.t);
            jy.push_back(sample.v);
        }

        std::ofstream file(path);
        file << std::setw(1) << json << "\n";
    }
}

arb::cable_cell gj_cell(cell_gid_type gid, unsigned ncell, double stim_duration) {
    // Create the sample tree that defines the morphology.
    arb::sample_tree tree;
    double soma_rad = 22.360679775/2.0; // convert diameter to radius in μm
    tree.append({{0,0,0,soma_rad}, 1}); // soma is a single sample point
    double dend_rad = 3./2;
    tree.append(0, {{0,0,soma_rad,     dend_rad}, 3});  // proximal point of the dendrite
    tree.append(1, {{0,0,soma_rad+300, dend_rad}, 3});  // distal end of the dendrite

    // Create a label dictionary that creates a single region that covers the whole cell.
    arb::label_dict d;
    d.set("all",  arb::reg::all());

    // Create the cell and set its electrical properties.
    arb::cable_cell cell(tree, d);
    cell.default_parameters.axial_resistivity = 100;       // [Ω·cm]
    cell.default_parameters.membrane_capacitance = 0.018;  // [F/m²]

    // Define the density channels and their parameters.
    arb::mechanism_desc nax("nax");
    nax["gbar"] = 0.04;
    nax["sh"] = 10;

    arb::mechanism_desc kdrmt("kdrmt");
    kdrmt["gbar"] = 0.0001;

    arb::mechanism_desc kamt("kamt");
    kamt["gbar"] = 0.004;

    arb::mechanism_desc pas("pas");
    pas["g"] =  1.0/12000.0;
    pas["e"] =  -65;

    // Paint density channels on all parts of the cell
    cell.paint("all", nax);
    cell.paint("all", kdrmt);
    cell.paint("all", kamt);
    cell.paint("all", pas);

    // Add a spike detector to the soma.
    cell.place(arb::mlocation{0,0}, arb::threshold_detector{10});

    // Add two gap junction sites.
    cell.place(arb::mlocation{0, 1}, arb::gap_junction_site{});
    cell.place(arb::mlocation{1, 1}, arb::gap_junction_site{});

    // Attach a stimulus to the second cell.
    if (!gid) {
        arb::i_clamp stim(0, stim_duration, 0.4);
        cell.place(arb::mlocation{0, 0.5}, stim);
    }

    // Add a synapse to the mid point of the first dendrite.
    cell.place(arb::mlocation{1, 0.5}, "expsyn");

    return cell;
}

gap_params read_options(int argc, char** argv) {
    using sup::param_from_json;

    gap_params params;
    if (argc<2) {
        std::cout << "Using default parameters.\n";
        return params;
    }
    if (argc>2) {
        throw std::runtime_error("More than command line one option not permitted.");
    }

    std::string fname = argv[1];
    std::cout << "Loading parameters from file: " << fname << "\n";
    std::ifstream f(fname);

    if (!f.good()) {
        throw std::runtime_error("Unable to open input parameter file: "+fname);
    }

    nlohmann::json json;
    json << f;

    param_from_json(params.name, "name", json);
    param_from_json(params.n_cables, "n-cables", json);
    param_from_json(params.n_cells_per_cable, "n-cells-per-cable", json);
    param_from_json(params.stim_duration, "stim-duration", json);
    param_from_json(params.event_min_delay, "event-min-delay", json);
    param_from_json(params.event_weight, "event-weight", json);
    param_from_json(params.sim_duration, "sim-duration", json);
    param_from_json(params.print_all, "print-all", json);

    if (!json.empty()) {
        for (auto it=json.begin(); it!=json.end(); ++it) {
            std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
        }
        std::cout << "\n";
    }

    return params;
}
