#include <any>
#include <iostream>
#include <iomanip>
#include <unordered_map>

#include <arborio/label_parse.hpp>

#include <arbor/spike_source_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/simulation.hpp>
#include <arbor/util/any_cast.hpp>
#include <arbor/util/any_ptr.hpp>

using namespace arborio::literals;

// Fan out network with n members
// - one spike source at gid 0
// - n-1 passive, soma-only cable cells
// Starts out disconnected, but new connections from the source to the cable cells may be added.
struct recipe: public arb::recipe {
    const std::string                           // Textuals labels for
        syn = "synapse",                        //     exponential synapse on cable cells
        det = "detector",                       //     spike detector on cable cells
        src = "src";                            //     spike source
    const double                                // Parameters
        r_soma  = 12.6157/2.0,                  //     soma radius; A=500 μm²
        f_spike = 0.25,                         //     spike source interval
        weight  = 5.5,                          //     connection weight
        delay   = 0.05;                         //                delay
    arb::locset center = "(location 0 0.5)"_ls; // Soma center
    arb::region all    = "(all)"_reg;           // Whole cell
    arb::cell_size_type n_ = 0;                 // Cell count

    std::unordered_map<arb::cell_gid_type, std::vector<arb::cell_connection>> connected; // lookup table for connections
    // Required but uninteresting methods
    recipe(arb::cell_size_type n): n_{n} {}
    arb::cell_size_type num_cells() const override { return n_; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override { return 0 == gid ? arb::cell_kind::spike_source : arb::cell_kind::cable; }
    std::any get_global_properties(arb::cell_kind kind) const override {
        if (kind == arb::cell_kind::spike_source) return {};
        arb::cable_cell_global_properties ccp;
        ccp.default_parameters = arb::neuron_parameter_defaults;
        return ccp;
    }
    // For printing Um
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type gid) const override {
        if (gid == 0) return {};
        return {arb::cable_probe_membrane_voltage{center}};
    }
    // Look up the (potential) connection to this cell
    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override {
        if (auto it = connected.find(gid); it != connected.end()) {
            return it->second;
        }
        return {};
    }
    // Connect cell `to` to the spike source
    void add_connection(arb::cell_gid_type to) { assert(to > 0); connected[to] = {arb::cell_connection({0, src}, {syn}, weight, delay)}; }
    // Return the cell at gid
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        // source at gid 0
        if (gid == 0) return arb::spike_source_cell{src, arb::regular_schedule(f_spike)};
        // all others are receiving cable cells; single CV w/ HH
        arb::segment_tree tree; tree.append(arb::mnpos, {-r_soma, 0, 0, r_soma}, {r_soma, 0, 0, r_soma}, 1);
        auto decor = arb::decor{}
            .paint(all, arb::density("hh", {{"gl", 5}}))
            .place(center, arb::synapse("expsyn"), syn)
            .place(center, arb::threshold_detector{-10.0}, det)
            .set_default(arb::cv_policy_every_segment());
        return arb::cable_cell({tree}, decor);
    }
};

// For demonstration: Avoid interleaving std::cout in multi-threaded scenarios.
// NEVER do this in HPC!!!
std::mutex mtx;

void sampler(arb::probe_metadata pm, std::size_t n, const arb::sample_record* samples) {
    auto* loc = arb::util::any_cast<const arb::mlocation*>(pm.meta);

    for (std::size_t i = 0; i<n; ++i) {
        std::lock_guard<std::mutex> lock{mtx};
        auto* value = arb::util::any_cast<const double*>(samples[i].data);
        std::cout << std::fixed << std::setprecision(4)
                  << "|  " << samples[i].time << " |      " << loc->pos << " | " << *value << " |\n";
    }
}

void spike_cb(const std::vector<arb::spike>& spikes) {
    for(const auto& spike: spikes) {
        std::lock_guard<std::mutex> lock{mtx};
        std::cout << " * " << spike.source << "@" << spike.time << '\n';
    }
}

void print_header(double from, double to) {
    std::lock_guard<std::mutex> lock{mtx};
    std::cout << "\n"
              << "Running simulation from " << from << "ms to " << to << "ms\n"
              << "Spikes are marked: *\n"
              << "\n"
              << "| Time/ms | Position/um | Um/mV    |\n"
              << "|---------+-------------+----------|\n";
}

const double dt = 0.05;

int main(int argc, char** argv) {
    auto rec = recipe(3);
    rec.add_connection(1);
    auto ctx = arb::make_context(arb::proc_allocation{8, -1});
    auto sim = arb::simulation(rec, ctx);
    sim.add_sampler(arb::all_probes, arb::regular_schedule(dt), sampler, arb::sampling_policy::exact);
    sim.set_global_spike_callback(spike_cb);
    print_header(0, 1);
    sim.run(1.0, dt);
    rec.add_connection(2);
    print_header(1, 2);
    sim.run(2.0, dt);
}
