#include <iostream>
#include <iomanip>
#include <vector>

#include <arbor/load_balance.hpp>
#include <arbor/adex_cell.hpp>
#include <arbor/simulation.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_cast.hpp>
#include <arbor/util/any_ptr.hpp>

namespace U = arb::units;
using namespace U::literals;

#include <tinyopt/tinyopt.h>

struct options {
    double t_end = 100.0;
    double dt = 0.025;
    float syn_weight = 0.01;
};

options parse_options(int argc, char** argv);


std::mutex mex;
std::vector<double> times;
std::vector<double> Um;
std::vector<double> w;

void sampler(const arb::probe_metadata& pm, const arb::sample_records& recs) {
    if (pm.id.tag == "Um") {
        using meta_t = arb::adex_probe_voltage::meta_type;
        auto reader = arb::sample_reader<meta_t>(pm.meta, recs);
        for (std::size_t ix = 0; ix < reader.n_row(); ++ix) {
            auto time = reader.time(ix);
            auto value = reader.value(ix);
            times.push_back(time);
            Um.push_back(value);            
        }
    }
    else if (pm.id.tag == "w") {
        using meta_t = arb::adex_probe_voltage::meta_type;
        auto reader = arb::sample_reader<meta_t>(pm.meta, recs);
        for (std::size_t ix = 0; ix < reader.n_row(); ++ix) {
            auto value = reader.value(ix);
            w.push_back(value);
        }        
    }
    else {
        std::cerr << "Unexpected tag '" << pm.id.tag << "'!\n";
    }    
}

void print() {
    std::cerr << std::fixed << std::setprecision(4);
    for (size_t ix = 0; ix < times.size(); ++ix) {
        std::cout << times[ix] << "," << Um[ix] << "," << w[ix] << '\n';
    }
}

struct recipe: public arb::recipe {
    arb::cell_size_type num_cells() const override { return 1; }

    std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override {
        return {arb::probe_info{arb::adex_probe_voltage{}, "Um"},
                arb::probe_info{arb::adex_probe_adaption{}, "w"}};
    }

    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::adex; }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        return arb::adex_cell{.source="src", .target="tgt"};
    }

    std::vector<arb::event_generator> event_generators(arb::cell_size_type) const override {
        return {arb::explicit_generator_from_milliseconds({"tgt"}, weight, std::vector{20.0, 28.0, 36.0, 44.0, 52.0, 60.0, 68.0, 76.0})};
    }

    arb::arb_weight_type weight = 10;
};

int main(int argc, char** argv) {
    options opt = parse_options(argc, argv);
    recipe R;
    R.weight = opt.syn_weight;

    arb::simulation sim(R);

    sim.add_sampler(arb::all_probes,
                    arb::regular_schedule(opt.dt * U::ms),
                    sampler);
    sim.set_global_spike_callback([](const auto& spks) {
        for (const auto& spk: spks) {
            std::cerr << spk.time << ", " << spk.source.gid << ", " << spk.source.index << '\n';
        }
    });

    sim.run(opt.t_end * U::ms, opt.dt * U::ms);
    print();
}

options parse_options(int argc, char** argv) {
    options opt;
    for (int ix = 1; ix < argc; ++ix) {
        auto arg = argv + ix;
        if (auto dt = to::parse<double>(arg, "-d", "--dt")) {
            opt.dt = dt.value();
        }
        else if (auto t_end = to::parse<double>(arg, "-t", "--t-end")) {
            opt.t_end = t_end.value();
        }
        else if (auto weight = to::parse<float>(arg, "-w", "--weight")) {
            opt.syn_weight = weight.value();
        }
        else {
            to::usage(argv[0], "[-d|--dt TIME] [-t|--t-end TIME] [-w|--weight WEIGHT]");
            std::exit(1);
        }
    }
    return opt;
}
