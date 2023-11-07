#include <iostream>
#include <iomanip>
#include <vector>

#include <arbor/load_balance.hpp>
#include <arbor/adex_cell.hpp>
#include <arbor/simulation.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_cast.hpp>
#include <arbor/util/any_ptr.hpp>


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

void sampler(const arb::probe_metadata& pm, std::size_t n, const arb::sample_record* samples) {
    auto* meta = arb::util::any_cast<const arb::adex_probe_metadata*>(pm.meta);
    if(meta == nullptr) {
        std::cerr << "Hey metadata is not ADEX metadata!\n";
        throw std::runtime_error{"ADEX metadata type mismatch"};
    }

    if (pm.id.tag == "Um") {
        for (std::size_t ix = 0; ix < n; ++ix) {
            auto* value = arb::util::any_cast<const double*>(samples[ix].data);
            if(value == nullptr) {
                std::cerr << "Hey payload is not const double* at index " << ix << "\n";
                throw std::runtime_error{"ADEX payload type mismatch"};
            }
            times.push_back(samples[ix].time);
            Um.push_back(*value);
        }
    }
    else {
        for (std::size_t ix = 0; ix < n; ++ix) {
            auto* value = arb::util::any_cast<const double*>(samples[ix].data);
            if(value == nullptr) {
                std::cerr << "Hey payload is not const double* at index " << ix << "\n";
                throw std::runtime_error{"ADEX payload type mismatch"};
            }
            w.push_back(*value);
        }
    }
}

void print() {
    std::cerr << std::fixed << std::setprecision(4);
    for (int ix = 0; ix < times.size(); ++ix) {
        std::cout << times[ix] << ", " << Um[ix] << ", " << w[ix] << '\n';
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
        auto cell = arb::adex_cell{"src", "tgt"};
        cell.V_m = -80;
        cell.E_R = -90;
        return cell;
    }

    std::vector<arb::event_generator> event_generators(arb::cell_size_type) const override {
        return {arb::regular_generator({"tgt"}, 10, 20, 8, 80)};
    }
};

int main(int argc, char** argv) {
    options opt = parse_options(argc, argv);
    recipe R;

    arb::simulation sim(R);

    sim.add_sampler(arb::all_probes,
                    arb::regular_schedule(opt.dt),
                    sampler);
    sim.set_global_spike_callback([](const auto& spks) {
        for (const auto& spk: spks) {
            std::cerr << spk.time << ", " << spk.source.gid << ", " << spk.source.index << '\n';
        }
    });

    sim.run(opt.t_end, opt.dt);
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
