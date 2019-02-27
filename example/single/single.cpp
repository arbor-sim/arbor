#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arbor/load_balance.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/morphology.hpp>
#include <arbor/swcio.hpp>
#include <arbor/simulation.hpp>
#include <arbor/simple_sampler.hpp>

#include <sup/tinyopt.hpp>

struct options {
    std::string swc_file;
    double t_end = 20;
    double dt = 0.025;
    float syn_weight = 0.01;
};

options parse_options(int argc, char** argv);
arb::morphology default_morphology();
arb::morphology read_swc(const std::string& path);

struct single_recipe: public arb::recipe {
    explicit single_recipe(arb::morphology m): morpho(m) {}

    arb::cell_size_type num_cells() const override { return 1; }
    arb::cell_size_type num_probes(arb::cell_gid_type) const override { return 1; }
    arb::cell_size_type num_targets(arb::cell_gid_type) const override { return 1; }

    arb::probe_info get_probe(arb::cell_member_type probe_id) const override {
        arb::segment_location mid_soma = {0, 0.5};
        arb::cell_probe_address probe = {mid_soma, arb::cell_probe_address::membrane_voltage};

        return {probe_id, 0, probe};
    }

    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override {
        return arb::cell_kind::cable1d_neuron;
    }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        arb::mc_cell c = make_mc_cell(morpho);

        for (auto& segment: c.segments()) {
            if (segment->is_soma()) {
                segment->add_mechanism("hh");
            }
            else {
                segment->add_mechanism("pas");

                // Discretize via NEURON d-lambda rule.
                double dx = segment->length_constant(100.)*0.3;
                unsigned n = std::ceil(segment->as_cable()->length()/dx);
                segment->set_compartments(n);

            }
        }

        // Add synapse to last segment.
        arb::cell_lid_type last_segment = morpho.components()-1;
        arb::segment_location end_last_segment = { last_segment, 1. };
        c.add_synapse(end_last_segment, "exp2syn");

        return c;
    }

    arb::morphology morpho;
};

int main(int argc, char** argv) {
    try {
        options opt = parse_options(argc, argv);
        single_recipe R(opt.swc_file.empty()? default_morphology(): read_swc(opt.swc_file));

        auto context = arb::make_context();
        arb::simulation sim(R, arb::partition_load_balance(R, context), context);

        arb::trace_data<double> trace;
        sim.add_sampler(arb::all_probes, arb::regular_schedule(0.1), arb::make_simple_sampler(trace));

        arb::spike_event spike = {{0, 0}, 1., opt.syn_weight}; // target, time, weight.
        sim.inject_events({spike});

        sim.run(opt.t_end, opt.dt);

        std::cout << std::fixed << std::setprecision(4);
        for (auto entry: trace) {
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
        if (auto dt = parse_opt<double>(arg, 'd', "dt")) {
            opt.dt = dt.value();
        }
        else if (auto t_end = parse_opt<double>(arg, 't', "t-end")) {
            opt.t_end = t_end.value();
        }
        else if (auto weight = parse_opt<float>(arg, 'w', "weight")) {
            opt.syn_weight = weight.value();
        }
        else if (auto swc = parse_opt<std::string>(arg, 'm', "morphology")) {
            opt.swc_file = swc.value();
        }
        else {
            usage(argv[0], "[-m|--morphology SWCFILE] [-d|--dt TIME] [-t|--t-end TIME] [-w|--weight WEIGHT]");
            std::exit(1);
        }
    }
    return opt;
}

arb::morphology default_morphology() {
    arb::morphology m;
    m.soma = {0., 0., 0., 6.3};

    std::vector<arb::section_point> dendrite = {
        {6.3, 0., 0., 0.5},
        {206.3, 0., 0., 0.2}
    };
    m.add_section(dendrite, 0);

    return m;
}

arb::morphology read_swc(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("unable to open SWC file: "+path);

    return arb::swc_as_morphology(arb::parse_swc_file(f));
}
