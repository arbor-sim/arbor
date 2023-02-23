#include <any>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <tinyopt/tinyopt.h>

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
using namespace arb;

struct linear: public recipe {
     linear(double ext, double dx, double Xi, double beta): l{ext}, d{dx}, i{Xi}, b{beta} {
        gprop.default_parameters = neuron_parameter_defaults;
        gprop.default_parameters.discretization = cv_policy_max_extent{d};
        gprop.add_ion("bla", 1, 23, 42, 0, 0);
    }

    cell_size_type num_cells()                                   const override { return 1; }
    cell_kind get_cell_kind(cell_gid_type)                       const override { return cell_kind::cable; }
    std::any get_global_properties(cell_kind)                    const override { return gprop; }
    std::vector<probe_info> get_probes(cell_gid_type)            const override { return {cable_probe_ion_diff_concentration_cell{"na"}}; }
    std::vector<event_generator> event_generators(cell_gid_type) const override { return {explicit_generator({"Zap"}, 0.005, std::vector<float>{0.f})}; }
    util::unique_any get_cell_description(cell_gid_type)         const override {
        // Stick morphology
        // -----|-----
        segment_tree tree;
        tree.append(mnpos, { -l, 0, 0, 3}, {l, 0, 0, 3}, 1);
        tree.append(0, { -l, 0, 0, 3}, {l, 0, 0, 3}, 2);
        // Setup
        decor decor;
        decor.set_default(init_int_concentration{"na", i});
        decor.paint("(tag 1)"_reg, ion_diffusivity{"na", b});
        decor.place("(location 0 0.5)"_ls, synapse("inject/x=bla", {{"alpha", 200.0*l}}), "Zap");
        decor.paint("(all)"_reg, density("decay/x=bla"));
        return cable_cell({tree}, decor);
    }

    cable_cell_global_properties gprop;
    double l, d, i, b;
};

std::ofstream out;

void sampler(probe_metadata pm, std::size_t n, const sample_record* samples) {
    auto ptr = util::any_cast<const mcable_list*>(pm.meta);
    assert(ptr);
    auto n_cable = ptr->size();
    out << "time,prox,dist,Xd\n"
        << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i<n; ++i) {
        const auto& [val, _ig] = *util::any_cast<const cable_sample_range*>(samples[i].data);
        for (unsigned j = 0; j<n_cable; ++j) {
            mcable loc = (*ptr)[j];
            out << samples[i].time << ',' << loc.prox_pos << ',' << loc.dist_pos << ',' << val[j] << '\n';
        }
    }
    out << '\n';
}

struct opt_t {
    double L  = 30.0;
    double dx = 1.0;
    double T  = 1.0;
    double dt = 0.01;
    double ds = 0.1;
    double Xi = 0.0;
    double dX = 0.005;
    std::string out = "log.csv";
    int gpu = -1;
};

opt_t read_options(int argc, char** argv) {
    auto usage = "\n"
                 "  -t|--tfinal     [Length of the simulation period (1 ms)]\n"
                 "  -d|--dt         [Simulation time step (0.01 ms)]\n"
                 "  -s|--ds         [Sampling interval (0.1 ms)]\n"
                 "  -g|--gpu        [Use GPU id (-1); enabled if >=0]\n"
                 "  -l|--length     [Length of stick (30 um)]\n"
                 "  -x|--dx         [Discretisation (1 um)]\n"
                 "  -i|--Xi         [Initial Na concentration (0 mM)]\n"
                 "  -b|--beta       [Na diffusivity (0.005 m^2/s)]\n"
                 "  -o|--output     [Save samples (log.csv)]\n";
    auto help = [argv, &usage] { to::usage(argv[0], usage); };
    opt_t opt;
    to::option options[] = {{opt.T,                                "-t", "--tfinal"},
                            {opt.dt,                               "-d", "--dt"},
                            {opt.ds,                               "-s", "--ds"},
                            {opt.L,                                "-l", "--length"},
                            {opt.dx,                               "-x", "--dx"},
                            {opt.Xi,                               "-i", "--Xi"},
                            {opt.dX,                               "-b", "--beta"},
                            {opt.gpu,                              "-g", "--gpu"},
                            {opt.out,                              "-o", "--out"},
                            {to::action(help), to::flag, to::exit, "-h", "--help"}};
    if (!to::run(options, argc, argv+1)) return opt_t{};
    if (argv[1])          throw to::option_error("Unrecognized argument", argv[1]);
    if (opt.dt <= 0.0)    throw std::runtime_error("Time step must be positive!");
    if (opt.ds <= 0.0)    throw std::runtime_error("Sampling interval must be positive!");
    if (opt.ds <  opt.dt) throw std::runtime_error("Time step is greater than a sampling interval!");
    if (opt.T  <= opt.ds) throw std::runtime_error("Runtime is less than a sampling interval!");
    if (opt.dX <= 0.0)    throw std::runtime_error("Diffusivity must be positive!");
    return opt;
}

int main(int argc, char** argv) {
    auto O = read_options(argc, argv);
    out = std::ofstream{O.out};
    if (!out.good()) throw std::runtime_error("Could not open output file for writing.");
    auto C = make_context({1, O.gpu});
    auto R = linear{O.L, O.dx, O.Xi, O.dX};
    simulation S(R, C, partition_load_balance(R, C));
    S.add_sampler(all_probes, regular_schedule(O.ds), sampler);
    S.run(O.T, O.dt);
    out.close();
}
