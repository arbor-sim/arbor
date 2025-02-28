#include <any>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <atomic>

#include <fmt/format.h>

#include <arbor/common_types.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/simulation.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_cast.hpp>
#include <arbor/util/any_ptr.hpp>
#include <tinyopt/tinyopt.h>

// Simulate a cell modelled as a simple cable with HH dynamics,
// emitting the results of a user specified probe over time.
namespace U = arb::units;
using namespace arb::units::literals;

const char* help_msg =
    "[OPTION]... PROBE\n"
    "\n"
    " -d, --dt=TIME       set simulation dt to TIME [ms]\n"
    " -T, --until=TIME    simulate until TIME [ms]\n"
    " -n, --n-cv=N        discretize with N CVs\n"
    " -t, --sample=TIME   take a sample every TIME [ms]\n"
    " -x, --at=X          take sample at relative position X along cable or index of synapse\n"
    " -h, --help          print extended usage information and exit\n"
    "\n"
    "Simulate a simple 1 mm cable with HH dynamics, taking samples according\n"
    "to PROBE (see below). Unless otherwise specified, the simulation will\n"
    "use 10 CVs and run for 100 ms with a time step of 0.025 ms.\n"
    "\n"
    "Samples are by default taken every 1 ms; for probes that test a specific\n"
    "point on the cell, that point is 0.5 along the cable (i.e. 500 µm) unless\n"
    "specified with the --at option.\n"
    "\n"
    "PROBE is one of:\n"
    "\n"
    "    v           membrane potential [mV] at X\n"
    "    i_axial     axial (distal) current [nA] at X\n"
    "    j_ion       total ionic membrane current density [A/m²] at X\n"
    "    j_na        sodium ion membrane current density [A/m²] at X\n"
    "    j_k         potassium ion membrane current density [A/m²] at X\n"
    "    c_na        internal sodium concentration [mmol/L] at X\n"
    "    c_k         internal potassium concentration [mmol/L] at X\n"
    "    hh_m        HH state variable m at X\n"
    "    hh_h        HH state variable h at X\n"
    "    hh_n        HH state variable n at X\n"
    "    expsyn_g    expsyn state variable g at X"
    "\n"
    "where X is the relative position along the cable as described above, or else:\n"
    "\n"
    "    all_v       membrane potential [mV] in each CV\n"
    "    all_i_ion   total ionic membrane current [nA] in each CV\n"
    "    all_i_na    sodium ion membrane current [nA] in each CV\n"
    "    all_i_k     potassium ion membrane current [nA] in each CV\n"
    "    all_i       total membrane current [nA] in each CV\n"
    "    all_c_na    internal sodium concentration [mmol/L] in each CV\n"
    "    all_c_k     internal potassium concentration [mmol/L] in each CV\n"
    "    all_hh_m    HH state variable m in each CV\n"
    "    all_hh_h    HH state variable h in each CV\n"
    "    all_hh_n    HH state variable n in each CV\n"
    "    all_expsyn_g expsyn state variable g for all synapses\n";

enum probe_kind { invalid, state, point, cell };

struct options {
    double sim_end = 100.0;   // [ms]
    double sim_dt = 0.025;    // [ms]
    double sample_dt = 1.0;   // [ms]
    unsigned n_cv = 10;
    std::any probe_addr;
    std::string value_name;
    probe_kind kind = probe_kind::invalid;
};

bool parse_options(options&, int& argc, char** argv);

template<typename M>
std::string show_location(const M& where) {
    if constexpr (std::is_same_v<std::remove_cv_t<M>, arb::mcable>) {
        return fmt::format("(cable {:1d} {:3.1f} {:3.1f})", where.branch, where.prox_pos, where.dist_pos);
    }
    else if constexpr (std::is_same_v<std::remove_cv_t<M>, arb::mlocation>) {
        return fmt::format("(location {:1d} {:3.1f})", where.branch, where.pos);
    }
    else if constexpr (std::is_same_v<std::remove_cv_t<M>, arb::cable_probe_point_info>) {
        return where.target;
    }
    else {
        throw std::runtime_error{"Unexpected metadata type"};
    }
}

// Do this once
static std::atomic<int> printed_header = 0;

template<typename M>
void sampler(arb::probe_metadata pm, const arb::sample_records& samples) {
    auto reader = arb::make_sample_reader<M>(pm.meta, samples);
    // Print CSV header for sample output
    if (0 == printed_header.fetch_add(1)) {
        std::cout << fmt::format("t", "");
        for (std::size_t iy = 0; iy < reader.width; ++iy) std::cout << ", " << show_location(reader.get_metadata(iy));
        std::cout << '\n';
    }

    for (std::size_t ix = 0; ix < reader.n_sample; ++ix) {
        std::cout << fmt::format("{:7.3f}", reader.get_time(ix));
        for (unsigned iy = 0; iy < reader.width; ++iy) {
            std::cout << fmt::format(", {:-8.4f}", reader.get_value(ix, iy));
        }
        std::cout  << '\n';
    }
}


struct cable_recipe: public arb::recipe {
    arb::cable_cell_global_properties gprop;
    std::any probe_addr;

    explicit cable_recipe(std::any probe_addr, unsigned n_cv):
        probe_addr(std::move(probe_addr)) {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.default_parameters.discretization = arb::cv_policy_fixed_per_branch(n_cv);
    }

    arb::cell_size_type num_cells() const override { return 1; }
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override { return {{probe_addr, "probe"}}; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::cable; }
    std::any get_global_properties(arb::cell_kind) const override { return gprop; }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        const double length = 1000; // [µm]
        const double diam   = 1;    // [µm]

        arb::segment_tree tree;
        tree.append(arb::mnpos, {0, 0, 0, 0.5*diam}, {length, 0, 0, 0.5*diam}, 1);

        auto decor = arb::decor{}
            .paint(arb::reg::all(), arb::density("hh"))                         // HH mechanism over whole cell.
            .place(arb::mlocation{0, 0.0}, arb::i_clamp{1._nA},    "iclamp")    // Inject a 1 nA current indefinitely.
            .place(arb::mlocation{0, 0.0}, arb::synapse("expsyn"), "synapse1")  // a synapse
            .place(arb::mlocation{0, 0.5}, arb::synapse("expsyn"), "synapse2"); // another synapse
        return arb::cable_cell(tree, decor);
    }

    virtual std::vector<arb::event_generator> event_generators(arb::cell_gid_type) const override {
        return {arb::poisson_generator({"synapse1"}, .005, 0._ms, 0.1_kHz),
                arb::poisson_generator({"synapse2"}, .1,   0._ms, 0.1_kHz)};
    }
};

int main(int argc, char** argv) {
    try {
        options opt;
        if (!parse_options(opt, argc, argv)) return -1;

        cable_recipe R(opt.probe_addr, opt.n_cv);

        arb::simulation sim(R);

        switch (opt.kind) {
        case probe_kind::cell:
            sim.add_sampler(arb::all_probes,
                            arb::regular_schedule(opt.sample_dt*U::ms),
                            sampler<arb::cable_state_cell_meta_type>);
            break;
        case probe_kind::state:
            sim.add_sampler(arb::all_probes,
                            arb::regular_schedule(opt.sample_dt*U::ms),
                            sampler<arb::cable_state_meta_type>);
            break;
        case probe_kind::point:
            sim.add_sampler(arb::all_probes,
                            arb::regular_schedule(opt.sample_dt*U::ms),
                            sampler<arb::cable_point_meta_type>);
            break;
        default:
            std::cerr << "Invalid probe kind\n";
            return -1;
        }
        std::cout << "Samples for '" << opt.value_name << "'\n"
                  << std::string(opt.value_name.size() + 14, '=')
                  << "\n\n";
        sim.run(opt.sim_end*U::ms, opt.sim_dt*U::ms);
    }
    catch (to::option_error& e) {
        to::usage_error(argv[0], "[OPTIONS]... PROBE\nTry '--help' for more information.", e.what());
        return 1;
    }
    catch (std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        return -2;
    }
}

static auto any2loc(std::any a) -> arb::mlocation {
    auto pos = 0.5;
    try {
        pos = std::any_cast<double>(a);
    }
    catch (const std::bad_any_cast& e) {
        std::cerr << "Invalid position specification, using default\n";
    }
    return arb::mlocation { .branch=0, .pos=pos };
}

static auto any2syn(std::any a) -> arb::cell_tag_type {
    arb::cell_tag_type pos = "synapse1";
    try {
        pos = std::any_cast<arb::cell_tag_type>(a);
    }
    catch (const std::bad_any_cast& e) {
        std::cerr << "Invalid synapse specification, using default\n";
    }
    return pos;
}


bool parse_options(options& opt, int& argc, char** argv) {
    using std::get;
    using namespace to;

    auto do_help = [&]() { usage(argv[0], help_msg); };

    // Map probe argument to output variable name and a lambda that makes specific probe address from a location.
    using probe_spec_t = std::tuple<std::string, probe_kind, std::function<std::any(std::any)>>;
    std::pair<const char*, probe_spec_t> probe_tbl[] {
        // located probes
        {"v",            {"v",        probe_kind::state, [](std::any a) -> std::any { return arb::cable_probe_membrane_voltage{any2loc(a)}; }}},
        {"i_axial",      {"i_axial",  probe_kind::state, [](std::any a) -> std::any { return arb::cable_probe_axial_current{any2loc(a)}; }}},
        {"j_ion",        {"j_ion",    probe_kind::state, [](std::any a) -> std::any { return arb::cable_probe_total_ion_current_density{any2loc(a)}; }}},
        {"j_na",         {"j_na",     probe_kind::state, [](std::any a) -> std::any { return arb::cable_probe_ion_current_density{any2loc(a), "na"}; }}},
        {"j_k",          {"j_k",      probe_kind::state, [](std::any a) -> std::any { return arb::cable_probe_ion_current_density{any2loc(a), "k"}; }}},
        {"c_na",         {"c_na",     probe_kind::state, [](std::any a) -> std::any { return arb::cable_probe_ion_int_concentration{any2loc(a), "na"}; }}},
        {"c_k",          {"c_k",      probe_kind::state, [](std::any a) -> std::any { return arb::cable_probe_ion_int_concentration{any2loc(a), "k"}; }}},
        {"hh_m",         {"hh_m",     probe_kind::state, [](std::any a) -> std::any { return arb::cable_probe_density_state{any2loc(a), "hh", "m"}; }}},
        {"hh_h",         {"hh_h",     probe_kind::state, [](std::any a) -> std::any { return arb::cable_probe_density_state{any2loc(a), "hh", "h"}; }}},
        {"hh_n",         {"hh_n",     probe_kind::state, [](std::any a) -> std::any { return arb::cable_probe_density_state{any2loc(a), "hh", "n"}; }}},
        {"expsyn_g",     {"expsyn_g", probe_kind::point, [](std::any a) -> std::any { return arb::cable_probe_point_state{any2syn(a), "expsyn", "g"}; }}},
        // all-of-cell probes
        {"all_v",        {"v",        probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_membrane_voltage_cell{}; }}},
        {"all_i_ion",    {"i_ion",    probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_total_ion_current_cell{}; }}},
        {"all_i_na",     {"i_na",     probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_ion_current_cell{"na"}; }}},
        {"all_i_k",      {"i_k",      probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_ion_current_cell{"k"}; }}},
        {"all_i",        {"i",        probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_total_current_cell{}; }}},
        {"all_c_na",     {"c_na",     probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_ion_int_concentration_cell{"na"}; }}},
        {"all_c_k",      {"c_k",      probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_ion_int_concentration_cell{"k"}; }}},
        {"all_hh_m",     {"hh_m",     probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_density_state_cell{"hh", "m"}; }}},
        {"all_hh_h",     {"hh_h",     probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_density_state_cell{"hh", "h"}; }}},
        {"all_hh_n",     {"hh_n",     probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_density_state_cell{"hh", "n"}; }}},
        {"all_expsyn_g", {"expsyn_g", probe_kind::cell, [](std::any) -> std::any { return arb::cable_probe_point_state_cell{"expsyn", "g"}; }}},
    };

    auto double_or_string = [](const char* arg) -> to::maybe<std::any> {
         try {
             return {{std::stod(arg)}};
         }
         catch (const std::exception& e) {
             return {{std::string(arg)}};
         }
    };

    probe_spec_t probe_spec;
    std::any p_pos;
    to::option cli_opts[] = {
        { to::action(do_help), to::flag, to::exit, "-h", "--help" },
        { opt.sim_dt,                              "-d", "--dt" },
        { opt.sim_end,                             "-T", "--until" },
        { opt.sample_dt,                           "-t", "--sample" },
        { to::sink(p_pos, double_or_string),       "-x", "--at" },
        { opt.n_cv,                                "-n", "--n-cv" },
        { {probe_spec, to::keywords(probe_tbl)}, to::single },
    };

    const auto& [p_name, p_kind, p_addr] = probe_spec;

    if (!to::run(cli_opts, argc, argv+1)) return false;
    if (!p_addr) throw to::user_option_error("missing PROBE");
    if (argv[1]) throw to::user_option_error("unrecognized option");
    opt.value_name = p_name;
    opt.probe_addr = p_addr(p_pos);
    opt.kind       = p_kind;
    return true;
}

