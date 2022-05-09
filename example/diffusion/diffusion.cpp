#include <any>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <arborio/label_parse.hpp>

#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/simulation.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_cast.hpp>
#include <arbor/util/any_ptr.hpp>

using namespace arborio::literals;

struct recipe: public arb::recipe {
    recipe(double ext): l{ext} {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.default_parameters.discretization = arb::cv_policy_max_extent{ext};
    }

    arb::cell_size_type num_cells()                             const override { return 1; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type)            const override { return arb::cell_kind::cable; }
    std::any get_global_properties(arb::cell_kind)              const override { return gprop; }
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override { return {arb::cable_probe_ion_diff_concentration_cell{"na"}}; }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        // Stick morphology
        // O-----
        arb::segment_tree tree;
        auto p = arb::mnpos;
        p = tree.append(p, { -30, 0, 0, 3}, {30, 0, 0, 3}, 1);
        arb::morphology morph{tree};

        arb::decor decor;
        decor.set_default(arb::init_int_concentration{"na", 0.0});
        decor.set_default(arb::ion_diffusivity{"na", 0.005});
        decor.place("(location 0 0.5)"_ls, arb::synapse("inject/x=na", {{"alpha", 200.0*l}}), "Zap");
        return arb::cable_cell(morph, {}, decor);
    }

    std::vector<arb::event_generator> event_generators(arb::cell_gid_type gid) const override {
        return {arb::explicit_generator({{{"Zap"}, 0.0, 0.005}})};
    }

    arb::cable_cell_global_properties gprop;
    double l;
};

std::ofstream out;

void sampler(arb::probe_metadata pm, std::size_t n, const arb::sample_record* samples) {
    auto ptr = arb::util::any_cast<const arb::mcable_list*>(pm.meta);
    assert(ptr);
    auto n_cable = ptr->size();
    out << std::fixed << std::setprecision(4) << "time,prox,dist,Xd\n";
    for (std::size_t i = 0; i<n; ++i) {
        const auto& [val, _ig] = *arb::util::any_cast<const arb::cable_sample_range*>(samples[i].data);
        for (unsigned j = 0; j<n_cable; ++j) {
            arb::mcable loc = (*ptr)[j];
            out << samples[i].time << ',' << loc.prox_pos << ',' << loc.dist_pos << ',' << val[j] << '\n';
        }
    }
    out << '\n';
}

int main(int argc, char** argv) {
    auto ctx = arb::make_context();
    auto mk_sim = [&ctx](double d) {
        recipe R{d};
        arb::simulation sim(R, arb::partition_load_balance(R, ctx), ctx);
        sim.add_sampler(arb::all_probes, arb::regular_schedule(0.1), sampler);
        sim.run(1.0, 0.01);
    };
    out = std::ofstream{"log-dx_1.csv"};
    mk_sim(1);
    out.close();
    out = std::ofstream{"log-dx_2.csv"};
    mk_sim(2);
    out.close();
}
