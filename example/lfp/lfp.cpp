#include <any>
#include <cassert>
#include <vector>
#include <iostream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/util/any_cast.hpp>
#include <arbor/util/any_ptr.hpp>
#include <arbor/util/unique_any.hpp>

#include <arborio/label_parse.hpp>

using std::any;
using arb::util::any_cast;
using arb::util::any_ptr;
using arb::util::unique_any;
using arb::cell_gid_type;
using arb::cell_member_type;

using namespace arborio::literals;

// Recipe represents one cable cell with one synapse, together with probes for total trans-membrane current, membrane voltage,
// ionic current density, and synaptic conductance. A sequence of spikes are presented to the one synapse on the cell.

struct lfp_demo_recipe: public arb::recipe {
    explicit lfp_demo_recipe(arb::event_generator events): events_(std::move(events))
    {
        make_cell(); // initializes cell_ and synapse_location_.
    }

    arb::cell_size_type num_cells() const override { return 1; }

    std::vector<arb::probe_info> get_probes(cell_gid_type) const override {
        // Four probes:
        //   0. Total membrane current across cell.
        //   1. Voltage at synapse location.
        //   2. Total ionic current density at synapse location.
        //   3. Expsyn synapse conductance value.
        return {
            arb::cable_probe_total_current_cell{},
            arb::cable_probe_membrane_voltage{synapse_location_},
            arb::cable_probe_total_ion_current_density{synapse_location_},
            arb::cable_probe_point_state{0, "expsyn", "g"}};
    }

    arb::cell_kind get_cell_kind(cell_gid_type) const override {
        return arb::cell_kind::cable;
    }

    unique_any get_cell_description(cell_gid_type) const override {
        return cell_;
    }

    virtual std::vector<arb::event_generator> event_generators(cell_gid_type) const override {
        return {events_};
    }

    any get_global_properties(arb::cell_kind) const override {
        arb::cable_cell_global_properties gprop;
        gprop.default_parameters = arb::neuron_parameter_defaults;
        return gprop;
    }

private:
    arb::cable_cell cell_;
    arb::locset synapse_location_;
    arb::event_generator events_;

    void make_cell() {
        using namespace arb;

        segment_tree tree;
        // Soma, length 20 μm radius 10 μm, with SWC tag 1.
        auto soma_apex = tree.append(arb::mnpos, {0, 0, -10, 10}, {0, 0, 10, 10}, 1);
        // Apical dendrite, length 490 μm, radius 1 μm, with SWC tag 4.
        tree.append(soma_apex, {0, 0, 10, 1},  {0, 0, 500, 1}, 4);

        synapse_location_ = "(on-components 0.5 (tag 1))"_ls;
        auto dec = decor()
            // Use NEURON defaults for reversal potentials, ion concentrations etc., but override ra, cm.
            .set_default(axial_resistivity{100})     // [Ω·cm]
            .set_default(membrane_capacitance{0.01}) // [F/m²]
            // Twenty CVs per branch on the dendrites (tag 4).
            .set_default(cv_policy_fixed_per_branch(20, arb::reg::tagged(4)))
            // Add pas and hh mechanisms:
            .paint("(tag 1)"_reg, density("hh")) // (default parameters)
            .paint("(tag 4)"_reg, density("pas/e=-70.0"))
            // Add exponential synapse at centre of soma.
            .place(synapse_location_, synapse("expsyn", {{"e", 0}, {"tau", 2}}), "syn");
        cell_ = cable_cell(tree, dec);
    }
};

struct position { double x, y, z; };

struct lfp_sampler {
    lfp_sampler(const arb::place_pwlin& placement, const arb::mcable_list& cables, const std::vector<position>& electrodes, double sigma) {
        // Compute response coefficients for each electrode, given a set of cable-like sources.
        const unsigned n_electrode = electrodes.size();
        response.assign(n_electrode, std::vector<double>(cables.size()));

        std::vector<arb::mpoint> midpoints;
        std::transform(cables.begin(), cables.end(), std::back_inserter(midpoints),
            [&placement](const auto& c) { return placement.at({c.branch, 0.5*(c.prox_pos+c.dist_pos)}); });

        const double coef = 1/(4*M_PI*sigma); // [Ω·m]
        for (unsigned i = 0; i<n_electrode; ++i) {
            const position& e = electrodes[i];

            std::transform(midpoints.begin(), midpoints.end(), response[i].begin(),
                [coef, &e](auto p) {
                    p.x -= e.x;
                    p.y -= e.y;
                    p.z -= e.z;
                    double r = std::sqrt(p.x*p.x+p.y*p.y+p.z*p.z); // [μm]
                    return coef/r; // [MΩ]
                });
        }
    }

    // On receipt of a sequence of cell-wide current samples, apply response matrix and save results to lfp_voltage.
    arb::sampler_function callback() {
        return [this](arb::probe_metadata pm, std::size_t n, const arb::sample_record* samples) {
            std::vector<double> currents;
            lfp_voltage.resize(response.size());

            for (std::size_t i = 0; i<n; ++i) {
                lfp_time.push_back(samples[i].time);

                auto data_ptr = any_cast<const arb::cable_sample_range*>(samples[i].data);
                assert(data_ptr);

                for (unsigned j = 0; j<response.size(); ++j) {
                    lfp_voltage[j].push_back(std::inner_product(data_ptr->first, data_ptr->second, response[j].begin(), 0.));
                }
            }
        };
    }

    std::vector<double> lfp_time;
    std::vector<std::vector<double>> lfp_voltage; // [mV] (one vector per electrode)

private:
    std::vector<std::vector<double>> response; // [MΩ]
};

// JSON output helpers:

template <typename T, typename F>
struct as_json_array_wrap {
    const T& data;
    F fn;
    as_json_array_wrap(const T& data, const F& fn): data(data), fn(fn) {}

    friend std::ostream& operator<<(std::ostream& out, const as_json_array_wrap& a) {
        out << '[';
        bool first = true;
        for (auto& x: a.data) out << (!first? ", ": (first=false, "")) << a.fn(x);
        return out << ']';
    }
};

struct {
    template <typename F>
    auto operator()(const F& fn) const {
        return [&fn](const auto& data) { return as_json_array_wrap<decltype(data), F>(data, fn); };
    }

    auto operator()() const {
        return this->operator()([](const auto& x) { return x; });
    }
} as_json_array;

// Run simulation.

int main(int argc, char** argv) {
    // Configuration
    const double t_stop = 100;    // [ms]
    const double sample_dt = 0.1; // [ms]
    const double dt = 0.1;        // [ms]

    // Weight 0.005 μS, onset at t = 0 ms, mean frequency 0.1 kHz.
    auto events = arb::poisson_generator({"syn"}, .005, 0., 0.1, std::minstd_rand{});
    lfp_demo_recipe recipe(events);
    arb::simulation sim(recipe);

    std::vector<position> electrodes = {
        {30, 0, 0},
        {30, 0, 100}
    };

    arb::morphology cell_morphology = any_cast<arb::cable_cell>(recipe.get_cell_description(0)).morphology();
    arb::place_pwlin placed_cell(cell_morphology);

    auto probe0_metadata = sim.get_probe_metadata(cell_member_type{0, 0});
    assert(probe0_metadata.size()==1); // Should only be one probe associated with this id.
    arb::mcable_list current_cables = *any_cast<const arb::mcable_list*>(probe0_metadata.at(0).meta);

    lfp_sampler lfp(placed_cell, current_cables, electrodes, 3.0);

    auto sample_schedule = arb::regular_schedule(sample_dt);
    sim.add_sampler(arb::one_probe({0, 0}), sample_schedule, lfp.callback(), arb::sampling_policy::exact);

    arb::trace_vector<double, arb::mlocation> membrane_voltage;
    sim.add_sampler(arb::one_probe({0, 1}), sample_schedule, make_simple_sampler(membrane_voltage), arb::sampling_policy::exact);

    arb::trace_vector<double> ionic_current_density;
    sim.add_sampler(arb::one_probe({0, 2}), sample_schedule, make_simple_sampler(ionic_current_density), arb::sampling_policy::exact);

    arb::trace_vector<double> synapse_g;
    sim.add_sampler(arb::one_probe({0, 3}), sample_schedule, make_simple_sampler(synapse_g), arb::sampling_policy::exact);

    sim.run(t_stop, dt);

    // Output results in JSON format suitable for plotting by plot-lfp.py script.

    auto get_t = [](const auto& x) { return x.t; };
    auto get_v = [](const auto& x) { return x.v; };
    auto scale = [](double s) { return [s](const auto& x) { return x*s; }; };
    auto to_xz = [](const auto& p) { return std::array<double, 2>{p.x, p.z}; };

    // Compute synaptic current from synapse conductance and membrane potential.
    std::vector<double> syn_i;
    assert(synapse_g.get(0).size()==membrane_voltage.get(0).size());
    std::transform(synapse_g.get(0).begin(), synapse_g.get(0).end(), membrane_voltage.get(0).begin(), std::back_inserter(syn_i),
        [](arb::trace_entry<double> g, arb::trace_entry<double> v) {
            assert(g.t==v.t);
            return g.v*v.v;
        });

    // Collect points from 2-d morphology in vectors of arrays (x, z, radius), one per branch.
    // (This process will be simplified with improvements to the place_pwlin API.)
    std::vector<std::vector<std::array<double, 3>>> samples;
    for (unsigned branch = 0; branch<cell_morphology.num_branches(); ++branch) {
        samples.push_back({});
        for (auto& seg: cell_morphology.branch_segments(branch)) {
            samples.back().push_back(std::array<double, 3>{seg.prox.x, seg.prox.z, seg.prox.radius});
            samples.back().push_back(std::array<double, 3>{seg.dist.x, seg.dist.z, seg.dist.radius});
        }
    }

    auto probe_xz = to_xz(placed_cell.at(membrane_voltage.get(0).meta));
    std::vector<std::array<double, 2>> electrodes_xz;
    std::transform(electrodes.begin(), electrodes.end(), std::back_inserter(electrodes_xz), to_xz);

    std::cout <<
        "{\n"
        "\"morphology\": {\n"
        "\"unit\": \"μm\",\n"
        "\"samples\": " << as_json_array(as_json_array(as_json_array()))(samples) << ",\n"
        "\"probe\": " << as_json_array()(probe_xz) << ",\n"
        "\"electrodes\": " << as_json_array(as_json_array())(electrodes_xz) << "\n"
        "},\n"
        "\"extracellular potential\": {\n"
        "\"unit\": \"μV\",\n"
        "\"time\": " << as_json_array()(lfp.lfp_time) << ",\n"
        "\"values\": " << as_json_array(as_json_array(scale(1e3)))(lfp.lfp_voltage) << "\n"
        "},\n"
        "\"synaptic current\": {\n"
        "\"unit\": \"nA\",\n"
        "\"time\": "  << as_json_array(get_t)(synapse_g.get(0)) << ",\n"
        "\"value\": " << as_json_array()(syn_i) << "\n"
        "},\n"
        "\"membrane potential\": {\n"
        "\"unit\": \"mV\",\n"
        "\"time\": "  << as_json_array(get_t)(membrane_voltage.get(0)) << ",\n"
        "\"value\": " << as_json_array(get_v)(membrane_voltage.get(0)) << "\n"
        "},\n"
        "\"ionic current density\": {\n"
        "\"unit\": \"A/m²\",\n"
        "\"time\": "  << as_json_array(get_t)(ionic_current_density.get(0)) << ",\n"
        "\"value\": " << as_json_array(get_v)(ionic_current_density.get(0)) << "\n"
        "}\n"
        "}\n";
}
