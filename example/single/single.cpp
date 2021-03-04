#include <any>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/simulation.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/util/any_visitor.hpp>

#include <arborio/cableio.hpp>
#include <arborio/cableio_error.hpp>
#include <arborio/swcio.hpp>

#include <tinyopt/tinyopt.h>

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
    arb::cell_size_type num_targets(arb::cell_gid_type) const override { return 1; }

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

        arb::decor decor;

        // Add HH mechanism to soma, passive channels to dendrites.
        decor.paint("\"soma\"", arb::mechanism_desc("hh").set("gnabar", 0.12).set("gkbar", 0.036));
        decor.paint("\"dend\"", "pas");

        // Add synapse to last branch.

        arb::cell_lid_type last_branch = morpho.num_branches()-1;
        arb::mlocation end_last_branch = { last_branch, 1. };
        decor.place(end_last_branch, "exp2syn");

        auto cell = arb::cable_cell(morpho, dict, decor);
        arborio::write_s_expr(std::cout, cell);
        std::cout << std::endl << std::endl;
        return cell;
    }

    arb::morphology morpho;
    arb::cable_cell_global_properties gprop;
};
std::ostream& operator<<(std::ostream& out, const arb::cv_policy&) {
    return out;
};

int main(int argc, char** argv) {
    try {
        options opt = parse_options(argc, argv);
        single_recipe R(opt.swc_file.empty()? default_morphology(): read_swc(opt.swc_file), opt.policy);

        auto context = arb::make_context();
        arb::simulation sim(R, arb::partition_load_balance(R, context), context);

        // Attach a sampler to the probe described in the recipe, sampling every 0.1 ms.

        arb::trace_vector<double> traces;
        sim.add_sampler(arb::all_probes, arb::regular_schedule(0.1), arb::make_simple_sampler(traces));

        // Trigger the single synapse (target is gid 0, index 0) at t = 1 ms with
        // the given weight.

        arb::spike_event spike = {{0, 0}, 1., opt.syn_weight};
        sim.inject_events({spike});

//        sim.run(opt.t_end, opt.dt);

//        for (auto entry: traces.at(0)) {
//            std::cout << entry.t << ", " << entry.v << "\n";
//        }
        std::string s = "((place \n"
                        "  (location 0 1)\n"
                        "  (mechanism \"exp2syn\"))\n"
                        " (paint \n"
                        "  (region \"dend\")\n"
                        "  (mechanism \"pas\"))\n"
                        " (paint \n"
                        "  (region \"soma\")\n"
                        "  (mechanism \"hh\" \n"
                        "   (\"gkbar\" 0.036000)\n"
                        "   (\"gnabar\" 0.120000))))";
        if (auto v = arborio::parse_decor(s)) {
            for (const auto& a: v->paintings()) {
                std:: cout << "paint on " << a.first << " : ";
                std::visit([](auto&& t){std::cout << t;}, a.second);
                std::cout << std::endl;
            }
            for (const auto& a: v->placements()) {
                std:: cout << "place on " << a.first << " : ";
                std::visit([](auto&& t){std::cout << t;}, a.second);
                std::cout << std::endl;
            }
            for (const auto& a: v->defaults().serialize()) {
                std:: cout << "default : ";
                std::visit([](auto&& t){std::cout << t;}, a);
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        else {
            throw v.error();
        }

        s = "   ((region-def \"soma\" \n"
            "      (tag 1))\n"
            "    (region-def \"dend\" \n"
            "      (join \n"
            "        (join \n"
            "          (tag 3)\n"
            "          (tag 4))\n"
            "        (tag 42))))";
        if (auto v = arborio::parse_label_dict(s)) {
            for (const auto& a: v->locsets()) {
                std:: cout << "locset " << a.first << " : " << a.second << std::endl;
            }
            for (const auto& a: v->regions()) {
                std:: cout << "region " << a.first << " : " << a.second << std::endl;
            }
            std::cout << std::endl;
        }
        else {
            throw v.error();
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
        if (auto dt = parse<double>(arg, 'd', "dt")) {
            opt.dt = dt.value();
        }
        else if (auto t_end = parse<double>(arg, 't', "t-end")) {
            opt.t_end = t_end.value();
        }
        else if (auto weight = parse<float>(arg, 'w', "weight")) {
            opt.syn_weight = weight.value();
        }
        else if (auto swc = parse<std::string>(arg, 'm', "morphology")) {
            opt.swc_file = swc.value();
        }
        else if (auto nseg = parse<unsigned>(arg, 'n', "cv-per-branch")) {
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
