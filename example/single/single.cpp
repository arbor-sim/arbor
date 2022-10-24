#include <any>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arborio/label_parse.hpp>

#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/simulation.hpp>
#include <arbor/simple_sampler.hpp>

#include <arborio/swcio.hpp>

#include <tinyopt/tinyopt.h>

using namespace arborio::literals;

struct options {
    std::string swc_file;
    double t_end = 20;
    double dt = 0.025;
    float syn_weight = 0.01;
    arb::cv_policy policy = arb::default_cv_policy();
};

options parse_options(int argc, char** argv);
arb::morphology default_morphology();
arb::morphology read_swc(const std::string& path);

struct single_recipe: public arb::recipe {
    explicit single_recipe(arb::morphology m, arb::cv_policy pol): morpho(std::move(m)) {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.default_parameters.discretization = pol;
    }

    arb::cell_size_type num_cells() const override { return 1; }

    std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override {
        arb::mlocation mid_soma = {0, 0.5};
        arb::cable_probe_membrane_voltage probe = {mid_soma};

        // Probe info consists of a probe address and an optional tag, for use
        // by the sampler callback.

        return {arb::probe_info{probe, 0}};
    }

    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override {
        return arb::cell_kind::cable;
    }

    std::any get_global_properties(arb::cell_kind) const override {
        return gprop;
    }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        arb::label_dict dict;
        using arb::reg::tagged;
        dict.set("soma", tagged(1));
        dict.set("dend", join(tagged(3), tagged(4), tagged(42)));

        auto decor = arb::decor{}
            // Add HH mechanism to soma, passive channels to dendrites.
            .paint("soma"_lab, arb::density("hh"))
            .paint("dend"_lab, arb::density("pas"))
            // Add synapse to last branch.
            .place(arb::mlocation{ morpho.num_branches()-1, 1. }, arb::synapse("exp2syn"), "synapse");

        return arb::cable_cell(morpho, decor, dict);
    }

    arb::morphology morpho;
    arb::cable_cell_global_properties gprop;
};

int main(int argc, char** argv) {
    try {
        options opt = parse_options(argc, argv);
        single_recipe R(opt.swc_file.empty()? default_morphology(): read_swc(opt.swc_file), opt.policy);

        arb::simulation sim(R);

        // Attach a sampler to the probe described in the recipe, sampling every 0.1 ms.

        arb::trace_vector<double> traces;
        sim.add_sampler(arb::all_probes, arb::regular_schedule(0.1), arb::make_simple_sampler(traces));

        // Trigger the single synapse (target is gid 0, index 0) at t = 1 ms with
        // the given weight.

        arb::spike_event spike = {0, 1., opt.syn_weight};
        arb::cell_spike_events cell_spikes = {0, {spike}};
        sim.inject_events({cell_spikes});

        sim.run(opt.t_end, opt.dt);

        for (auto entry: traces.at(0)) {
            std::cout << entry.t << ", " << entry.v << "\n";
        }
    }
    catch (std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        return 2;
    }
}

options parse_options(int argc, char** argv) {
    using namespace to;
    options opt;

    char** arg = argv+1;
    while (*arg) {
        if (auto dt = parse<double>(arg, "-d", "--dt")) {
            opt.dt = dt.value();
        }
        else if (auto t_end = parse<double>(arg, "-t", "--t-end")) {
            opt.t_end = t_end.value();
        }
        else if (auto weight = parse<float>(arg, "-w", "--weight")) {
            opt.syn_weight = weight.value();
        }
        else if (auto swc = parse<std::string>(arg, "-m", "--morphology")) {
            opt.swc_file = swc.value();
        }
        else if (auto nseg = parse<unsigned>(arg, "-n", "--cv-per-branch")) {
            opt.policy = arb::cv_policy_fixed_per_branch(nseg.value());
        }
        else {
            usage(argv[0], "[-m|--morphology SWCFILE] [-d|--dt TIME] [-t|--t-end TIME] [-w|--weight WEIGHT] [-n|--cv-per-branch N]");
            std::exit(1);
        }
    }
    return opt;
}

// If no SWC file is given, the default morphology consists
// of a soma of radius 6.3 µm and a single unbranched dendrite
// of length 200 µm and radius decreasing linearly from 0.5 µm
// to 0.2 µm.

arb::morphology default_morphology() {
    arb::segment_tree tree;

    tree.append(arb::mnpos, { -6.3, 0.0, 0.0, 6.3}, {  6.3, 0.0, 0.0, 6.3}, 1);
    tree.append(         0, {  6.3, 0.0, 0.0, 0.5}, {206.3, 0.0, 0.0, 0.2}, 3);

    return arb::morphology(tree);
}

arb::morphology read_swc(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("unable to open SWC file: "+path);

    return arborio::load_swc_arbor(arborio::parse_swc(f));
}
