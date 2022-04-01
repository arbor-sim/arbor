#include <any>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    recipe() {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.default_parameters.discretization = arb::cv_policy_max_extent{3};
    }

    arb::cell_size_type num_cells()                             const override { return 1; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type)            const override { return arb::cell_kind::cable; }
    std::any get_global_properties(arb::cell_kind)              const override { return gprop; }
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override { return {arb::cable_probe_membrane_voltage_cell{}}; }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        arb::segment_tree tree;
        auto p = arb::mnpos;
        p = tree.append(p, {-3, 0, 0, 3}, { 3, 0, 0, 3}, 1);
        p = tree.append(p, { 3, 0, 0, 1}, {33, 0, 0, 1}, 3);
        arb::morphology morph{tree};

        arb::label_dict dict;
        dict.set("soma", arb::reg::tagged(1));
        dict.set("dend", arb::reg::tagged(3));

        arb::decor decor;
        decor.set_default(arb::init_int_concentration{"na", 0.2});
        decor.set_default(arb::ion_diffusivity{"na", 0.5});
        decor.paint("soma"_lab, arb::init_int_concentration{"na", 1.2});
        decor.paint("soma"_lab, arb::density("hh"));
        decor.paint("soma"_lab, arb::ion_diffusivity{"na", 0.005});
        decor.paint("dend"_lab, arb::density("pas"));

        return arb::cable_cell(morph, dict, decor);
    }

    arb::cable_cell_global_properties gprop;
};

void sampler(arb::probe_metadata pm, std::size_t n, const arb::sample_record* samples) {
    auto ptr = arb::util::any_cast<const arb::mcable_list*>(pm.meta);
    assert(ptr);
    auto n_cable = ptr->size();
    std::cout << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i<n; ++i) {
        auto* value_range = arb::util::any_cast<const arb::cable_sample_range*>(samples[i].data);
        assert(value_range);
        assert(n_cable==value_range->second-value_range->first);
        for (unsigned j = 0; j<n_cable; ++j) {
            arb::mcable where = (*ptr)[j];
            std::cout << samples[i].time << ", "
                      << where.prox_pos << ", "
                      << where.dist_pos << ", "
                      << value_range->first[j] << '\n';
        }
    }
}

int main(int argc, char** argv) {
    recipe R;
    auto context = arb::make_context();
    arb::simulation sim(R,
                        arb::partition_load_balance(R, context),
                        context);

    sim.add_sampler(arb::all_probes,
                    arb::regular_schedule(0.1),
                    sampler);

    sim.run(0.01, 0.005);
}
