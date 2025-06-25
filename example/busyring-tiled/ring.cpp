#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <array>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
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
#include <arbor/symmetric_recipe.hpp>

#include <arborenv/default_env.hpp>
#include <arborenv/gpu_env.hpp>

#include <arborio/label_parse.hpp>

#include "parameters.hpp"

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <arborenv/with_mpi.hpp>
#endif

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_kind;
using arb::time_type;
using arb::cable_probe_membrane_voltage;

using namespace arborio::literals;
namespace U = arb::units;

// Writes voltage trace as a json file.
void write_trace_json(std::string fname, const arb::trace_data<double>& trace);

// Generate a cell.
arb::cable_cell branch_cell(arb::cell_gid_type gid, const cell_parameters& params);
arb::cable_cell complex_cell(arb::cell_gid_type gid, const cell_parameters& params);

struct ring_recipe: public arb::recipe {
    ring_recipe(ring_params params):
        num_cells_(params.num_cells),
        min_delay_(params.min_delay),
        event_weight_(params.event_weight),
        params_(params)
    {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.catalogue.extend(arb::global_allen_catalogue());

        if (params.cell.complex_cell) {
            gprop.default_parameters.reversal_potential_method["ca"] = "nernst/ca";
            gprop.default_parameters.axial_resistivity = 100;
            gprop.default_parameters.temperature_K = 34 + 273.15;
            gprop.default_parameters.init_membrane_potential = -90;
        }
    }

    std::any get_global_properties(cell_kind kind) const override { return gprop; }
    cell_size_type num_cells() const override { return num_cells_; }
    cell_kind get_cell_kind(cell_gid_type gid) const override { return cell_kind::cable; }
    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        if (params_.cell.complex_cell) return complex_cell(gid, params_.cell);
        return branch_cell(gid, params_.cell);
    }

    // Each cell has one incoming connection, from cell with gid-1,
    // and fan_in-1 random connections with very low weight.
    arb::connection_list connections_on(cell_gid_type gid) const override {
        arb::connection_list cons;
        if (params_.resolve_sources) {
            const auto ncons = params_.cell.synapses;
            cons.reserve(ncons);

            const auto s = params_.ring_size;
            const auto group = gid/s;
            const auto group_start = s*group;
            const auto group_end = std::min(group_start+s, num_cells_);
            cell_gid_type src = gid==group_start? group_end-1: gid-1;
            cons.push_back({{src, "d"}, {"p"}, event_weight_, min_delay_*U::ms});
        }
        return cons;
    }

    // Each cell has one incoming connection, from cell with gid-1,
    // and fan_in-1 random connections with very low weight.
    arb::raw_connection_list raw_connections_on(cell_gid_type gid) const override {
        arb::raw_connection_list cons;
        if (!params_.resolve_sources) {
            const auto ncons = params_.cell.synapses;
            cons.reserve(ncons);

            const auto s = params_.ring_size;
            const auto group = gid/s;
            const auto group_start = s*group;
            const auto group_end = std::min(group_start+s, num_cells_);
            cell_gid_type src = gid==group_start? group_end-1: gid-1;
            cons.push_back({{src, 0}, {"p"}, event_weight_, min_delay_*U::ms});
        }
        return cons;
    }

    bool resolve_sources () const override { return params_.resolve_sources; }

    // Return one event generator on the first cell of each ring.
    // This generates a single event that will kick start the spiking on the sub-ring.
    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        if (gid%params_.ring_size == 0) return {arb::explicit_generator_from_milliseconds({"p"}, event_weight_, std::vector{1.0})};
        return {};
    }

    std::vector<arb::probe_info> get_probes(cell_gid_type gid) const override {
        // Measure at the soma.
        arb::mlocation loc{0, 0.0};
        return {{cable_probe_membrane_voltage{loc}, "Um"}};
    }

    cell_size_type num_cells_;
    double min_delay_;
    float event_weight_;
    ring_params params_;

    arb::cable_cell_global_properties gprop;
};

struct tiled_recipe: arb::recipe {
    tiled_recipe(ring_params ps, std::size_t nt): recipe_{std::move(ps)}, num_tiles_{nt} {}

    cell_size_type num_cells() const override { return recipe_.num_cells() * num_tiles_; }
    arb::util::unique_any get_cell_description(cell_gid_type gid) const override { return recipe_.get_cell_description(gid % recipe_.num_cells()); }
    arb::cell_kind get_cell_kind(cell_gid_type gid) const override { return recipe_.get_cell_kind(gid % recipe_.num_cells()); }
    std::any get_global_properties(arb::cell_kind k) const override { return recipe_.get_global_properties(k); }

    arb::connection_list connections_on(cell_gid_type gid) const override {
        if (!recipe_.resolve_sources()) return {};
        // tile internal
        auto conns = recipe_.connections_on(gid % recipe_.num_cells());
        auto ncons = recipe_.params_.cell.synapses;
        // Used to pick source cell for a connection.
        std::uniform_int_distribution<cell_gid_type> dist(0, num_tiles_*recipe_.num_cells_ - 2);
        // Used to pick delay for a connection.
        std::uniform_real_distribution<float> delay_dist(0, 2*recipe_.min_delay_);
        auto src_gen = std::mt19937(gid);
        for (unsigned i = 1; i < ncons; ++i) {
            auto src = dist(src_gen);
            if (src==gid) ++src;
            const float delay = recipe_.min_delay_ + delay_dist(src_gen);
            conns.push_back(arb::cell_connection({src, "d"}, {"p"}, 0.f, delay*U::ms));
        }
        return conns;
    }

    arb::raw_connection_list raw_connections_on(cell_gid_type gid) const override {
        if (recipe_.resolve_sources()) return {};
        // tile internal
        auto conns = recipe_.raw_connections_on(gid % recipe_.num_cells());
        auto ncons = recipe_.params_.cell.synapses;
        // Used to pick source cell for a connection.
        std::uniform_int_distribution<cell_gid_type> dist(0, num_tiles_*recipe_.num_cells_ - 2);
        // Used to pick delay for a connection.
        std::uniform_real_distribution<float> delay_dist(0, 2*recipe_.min_delay_);
        auto src_gen = std::mt19937(gid);
        for (unsigned i = 1; i < ncons; ++i) {
            auto src = dist(src_gen);
            if (src==gid) ++src;
            const float delay = recipe_.min_delay_ + delay_dist(src_gen);
            conns.push_back({{src, 0}, {"p"}, 0.f, delay*U::ms});
        }
        return conns;
    }

    bool resolve_sources() const override { return recipe_.resolve_sources(); }


    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override { return recipe_.event_generators(gid % recipe_.num_cells()); }

    ring_recipe recipe_;
    std::size_t num_tiles_;
};

int main(int argc, char** argv) {
    try {
        bool root = true;

        auto params = read_options(argc, argv);

        arb::proc_allocation resources;
        resources.num_threads = arbenv::default_concurrency();
        resources.bind_threads = params.bind_threads;

        auto ctx = arb::make_context(resources, arb::dry_run_info(params.num_tiles, params.num_cells));

#ifdef ARB_PROFILE_ENABLED
        arb::profile::profiler_initialize(ctx);
#endif

        // Print a banner with information about hardware configuration
        if (root) {
            std::cout << "gpu:      " << (has_gpu(ctx)? "yes": "no") << "\n"
                      << "threads:  " << num_threads(ctx) << "\n"
                      << "mpi:      " << (has_mpi(ctx)? "yes": "no") << "\n"
                      << "ranks:    " << num_ranks(ctx) << "\n"
                      << std::endl;
        }

        arb::profile::meter_manager meters;
        meters.start(ctx);

        // Create an instance of our recipe.
        tiled_recipe recipe(params, params.num_tiles);
        auto decomp = arb::partition_load_balance(recipe, ctx, {{arb::cell_kind::cable, params.hint}});
        // Construct the model.
        arb::simulation sim(recipe, ctx, decomp);

        // Set up the probe that will measure voltage in the cell.
        meters.checkpoint("model-init", ctx);

        // Run the simulation.
        if (root) sim.set_epoch_callback(arb::epoch_progress_bar());
        if (root) std::cout << "running simulation\n" << std::endl;
        sim.run(params.duration*U::ms, params.dt*U::ms);

        meters.checkpoint("model-run", ctx);

        auto ns = sim.num_spikes();
        auto report = arb::profile::make_meter_report(meters, ctx);

        // Write spikes to file
        if (root) {
            std::cout << "\n" << ns << " spikes generated at rate of "
                      << params.duration/ns << " ms between spikes\n"
                      << report << '\n';
        }
#ifdef ARB_PROFILE_ENABLED
        // if (root) arb::profile::print_profiler_summary(std::cout, 0);
#endif
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in ring miniapp: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

void write_trace_json(std::string fname, const arb::trace_data<double>& trace) {
    nlohmann::json json;
    json["name"] = "ring demo";
    json["units"] = "mV";
    json["cell"] = "0.0";
    json["probe"] = "0";

    auto& jt = json["data"]["time"];
    auto& jy = json["data"]["voltage"];

    for (const auto& sample: trace) {
        jt.push_back(sample.t);
        jy.push_back(sample.v);
    }

    std::ofstream file(fname);
    file << std::setw(1) << json << "\n";
}

// Helper used to interpolate in branch_cell.
template <typename T>
double interp(const std::array<T,2>& r, unsigned i, unsigned n) {
    double p = i * 1./(n-1);
    double r0 = r[0];
    double r1 = r[1];
    return r[0] + p*(r1-r0);
}

arb::segment_tree generate_morphology(arb::cell_gid_type gid, const cell_parameters& params) {
    arb::segment_tree tree;

    double soma_radius = 12.6157/2.0;
    int soma_tag = 1;
    tree.append(arb::mnpos, {0, 0,-soma_radius, soma_radius}, {0, 0, soma_radius, soma_radius}, soma_tag); // For area of 500 μm².

    std::vector<std::vector<unsigned>> levels;
    levels.push_back({0});

    // Standard mersenne_twister_engine seeded with gid.
    std::mt19937 gen(gid);
    std::uniform_real_distribution<double> dis(0, 1);

    double dend_radius = 0.5; // Diameter of 1 μm for each cable.
    int dend_tag = 3;

    double dist_from_soma = soma_radius;
    for (unsigned i=0; i<params.max_depth; ++i) {
        // Branch prob at this level.
        double bp = interp(params.branch_probs, i, params.max_depth);
        // Length at this level.
        double l = interp(params.lengths, i, params.max_depth);
        // Number of compartments at this level.
        unsigned nc = std::round(interp(params.compartments, i, params.max_depth));

        std::vector<unsigned> sec_ids;
        for (unsigned sec: levels[i]) {
            for (unsigned j=0; j<2; ++j) {
                if (dis(gen)<bp) {
                    auto z = dist_from_soma;
                    auto dz = l/nc;
                    auto p = sec;
                    for (unsigned k=1; k<nc; ++k) {
                        p = tree.append(p, {0,0,z+(k+1)*dz, dend_radius}, dend_tag);
                    }
                    sec_ids.push_back(p);
                }
            }
        }
        if (sec_ids.empty()) {
            break;
        }
        levels.push_back(sec_ids);

        dist_from_soma += l;
    }

    return tree;
}

arb::cable_cell complex_cell(arb::cell_gid_type gid, const cell_parameters& params) {
    using arb::reg::tagged;
    using arb::reg::all;
    using arb::ls::location;
    using arb::ls::uniform;

    arb::segment_tree tree = generate_morphology(gid, params);

    auto rall  = arb::reg::all();
    auto soma = tagged(1);
    auto axon = tagged(2);
    auto dend = tagged(3);
    auto apic = tagged(4);
    auto cntr = location(0, 0.5);
    auto syns = arb::ls::uniform(rall, 0, params.synapses-2, gid);

    arb::decor decor;

    decor.paint(rall, arb::init_reversal_potential{"k",  -107.0*U::mV});
    decor.paint(rall, arb::init_reversal_potential{"na",   53.0*U::mV});

    decor.paint(soma, arb::axial_resistivity{133.577*U::Ohm*U::cm});
    decor.paint(soma, arb::membrane_capacitance{4.21567e-2*U::F/U::m2});

    decor.paint(soma, arb::density("pas/e=-76.4024", {{"g", 0.000119174}}));
    decor.paint(soma, arb::density("NaV",            {{"gbar", 0.0499779}}));
    decor.paint(soma, arb::density("SK",             {{"gbar", 0.000733676}}));
    decor.paint(soma, arb::density("Kv3_1",          {{"gbar", 0.186718}}));
    decor.paint(soma, arb::density("Ca_HVA",         {{"gbar", 9.96973e-05}}));
    decor.paint(soma, arb::density("Ca_LVA",         {{"gbar", 0.00344495}}));
    decor.paint(soma, arb::density("CaDynamics",     {{"gamma", 0.0177038}, {"decay", 42.2507}}));
    decor.paint(soma, arb::density("Ih",             {{"gbar", 1.07608e-07}}));

    decor.paint(dend, arb::axial_resistivity{68.355*U::Ohm*U::cm});
    decor.paint(dend, arb::membrane_capacitance{2.11248e-2*U::F/U::m2});

    decor.paint(dend, arb::density("pas/e=-88.2554", {{"g", 9.57001e-05}}));
    decor.paint(dend, arb::density("NaV",            {{"gbar", 0.0472215}}));
    decor.paint(dend, arb::density("Kv3_1",          {{"gbar", 0.186859}}));
    decor.paint(dend, arb::density("Im_v2",          {{"gbar", 0.00132163}}));
    decor.paint(dend, arb::density("Ih",             {{"gbar", 9.18815e-06}}));

    decor.place(cntr, arb::threshold_detector{-20.0*U::mV}, "d");
    decor.place(cntr, arb::synapse("expsyn"), "p");

    if (params.synapses>1) decor.place(syns, arb::synapse("expsyn"), "s");

    return {arb::morphology(tree), decor, {}, arb::cv_policy_every_segment()};
}

arb::cable_cell branch_cell(arb::cell_gid_type gid, const cell_parameters& params) {
    using arb::reg::tagged;

    arb::segment_tree tree = generate_morphology(gid, params);

    auto soma = tagged(1);
    auto dnds = join(tagged(3), tagged(4));
    auto syns = arb::ls::uniform(arb::reg::all(), 0, params.synapses-2, gid);

    arb::decor decor;

    decor.paint(soma, arb::density{"hh"});
    decor.paint(dnds, arb::density{"pas"});
    decor.set_default(arb::axial_resistivity{100*U::Ohm*U::cm}); // [Ω·cm]

    // Add spike threshold detector at the soma.
    decor.place(arb::mlocation{0,0}, arb::threshold_detector{10*U::mV}, "d");

    // Add a synapse to proximal end of first dendrite.
    decor.place(arb::mlocation{1, 0}, arb::synapse{"expsyn"}, "p");

    // Add additional synapses that will not be connected to anything.
    if (params.synapses>1) {
        decor.place(syns, arb::synapse{"expsyn"}, "s");
    }

    // Make a CV between every sample in the sample tree.
    return {arb::morphology(tree), decor, {}, arb::cv_policy_every_segment()};
}
