/*
 * Neighbourhood spike exchange.
 *
 * Every rank holds a tile of LIF cells; the ranks form a ring, which is split
 * into levels by distance. With --widths 1,4 there are
 *   - the own rank,
 *   - level 1: the ranks 1 step away on the ring,
 *   - level 2: the ranks 2 to 4 steps away,
 *   - far: all other ranks.
 * Each cell receives --in-degrees connections from cells in each of these
 * groups, with the respective --delays; delays should grow with distance.
 *
 * The model is simulated twice: with the default exchange, which runs a global
 * all-to-all every epoch (half the shortest delay), and with neighbourhood
 * exchange. There, each level is exchanged with a sparse MPI neighbourhood
 * all-gather and the far ranks with a global all-to-all, each only as often as
 * the delays from that group require.
 *
 * Run with MPI, e.g. `mpirun -n 16 ./neighborhood --widths 1,4`, or emulate
 * many ranks with `./neighborhood --dry-run 64`.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tinyopt/tinyopt.h>

#include <arbor/context.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/simulation.hpp>
#include <arbor/spike_exchange.hpp>
#include <arbor/symmetric_recipe.hpp>
#include <arbor/version.hpp>

#include <sup/ioutil.hpp>

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <arborenv/with_mpi.hpp>
#endif

namespace U = arb::units;

struct options {
    unsigned cells_per_rank = 1000;
    // Outer ring distance of each level, increasing.
    std::vector<unsigned> widths = {1, 4};
    // Per group (own rank, levels, far): connections per cell and their delay [ms].
    std::vector<unsigned> in_degrees = {40, 30, 20, 10};
    std::vector<double> delays = {1, 2, 5, 20};
    double tfinal = 200;           // [ms]
    double dt = 0.025;             // [ms]
    unsigned dry_run_ranks = 0;    // emulate this many ranks; 0 for a real run
};

std::optional<options> parse_options(int argc, char** argv);

// Distance between two ranks on the ring.
unsigned ring_distance(unsigned a, unsigned b, unsigned num_ranks) {
    auto d = a > b? a - b: b - a;
    return std::min(d, num_ranks - d);
}

// Group of rank b as seen from rank a: 0 for the own rank, k for level k, levels+1 for far.
unsigned group_of(unsigned a, unsigned b, unsigned num_ranks, const std::vector<unsigned>& widths) {
    auto d = ring_distance(a, b, num_ranks);
    if (d == 0) return 0;
    auto level = std::lower_bound(widths.begin(), widths.end(), d) - widths.begin();
    return level + 1;
}

// The tile of rank 0; symmetric_recipe shifts it to the other ranks.
class ring_tile: public arb::tile {
public:
    ring_tile(const options& opt, unsigned num_ranks):
        opt_(opt), num_ranks_(num_ranks), ranks_by_group_(opt.widths.size() + 2) {
        for (unsigned r = 0; r < num_ranks; ++r) ranks_by_group_[group_of(0, r, num_ranks, opt.widths)].push_back(r);
    }

    arb::cell_size_type num_cells() const override { return opt_.cells_per_rank; }
    arb::cell_size_type num_tiles() const override { return num_ranks_; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::lif; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        return arb::lif_cell{.source="src", .target="tgt"};
    }

    // Sources are given relative to rank 0.
    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override {
        std::mt19937 rng(gid);
        std::uniform_int_distribution<unsigned> any_cell(0, opt_.cells_per_rank - 1);
        std::vector<arb::cell_connection> conns;
        auto connect = [&](unsigned rank, double delay) {
            // One in four connections is inhibitory.
            float weight = conns.size()%4? 20.f: -60.f;
            conns.emplace_back(arb::cell_global_label_type{rank*opt_.cells_per_rank + any_cell(rng), "src"},
                               arb::cell_local_label_type{"tgt"},
                               weight, delay*U::ms);
        };

        for (std::size_t g = 0; g < ranks_by_group_.size(); ++g) {
            // Groups can be empty if the ring is small.
            const auto& ranks = ranks_by_group_[g];
            if (ranks.empty()) continue;
            std::uniform_int_distribution<std::size_t> any_rank(0, ranks.size() - 1);
            for (unsigned i = 0; i < opt_.in_degrees[g]; ++i) connect(ranks[any_rank(rng)], opt_.delays[g]);
        }
        return conns;
    }

    // Called with the global gid.
    std::vector<arb::event_generator> event_generators(arb::cell_gid_type gid) const override {
        return {arb::poisson_generator({"tgt"}, 30.f, 0*U::ms, 0.8*U::kHz, gid)};
    }

private:
    options opt_;
    unsigned num_ranks_;
    std::vector<std::vector<unsigned>> ranks_by_group_;
};

// The ranks this rank receives from, one neighbourhood per level.
arb::neighborhood_spike_exchange ring_neighborhoods(unsigned rank, unsigned num_ranks, const std::vector<unsigned>& widths) {
    // Every rank must pass all levels, even those that are empty on a small ring.
    std::vector<std::vector<int>> levels(widths.size());
    for (unsigned r = 0; r < num_ranks; ++r) {
        auto g = group_of(rank, r, num_ranks, widths);
        if (g >= 1 && g <= widths.size()) levels[g - 1].push_back(r);
    }
    return {levels};
}

struct result {
    std::size_t num_spikes;
    double init_time;
    double run_time;
};

result simulate(const arb::recipe& rec, arb::context ctx, const std::optional<arb::neighborhood_spike_exchange>& exchange, const options& opt) {
    using clock = std::chrono::steady_clock;
    auto seconds = [](clock::duration d) { return std::chrono::duration<double>(d).count(); };

    auto t0 = clock::now();
    auto sim = arb::simulation::create(rec).set_context(ctx);
    if (exchange) sim.set_spike_exchange(*exchange);
    auto simulation = sim.make_unique();
    auto t1 = clock::now();
    simulation->run(opt.tfinal*U::ms, opt.dt*U::ms);
    auto t2 = clock::now();

    return {simulation->num_spikes(), seconds(t1 - t0), seconds(t2 - t1)};
}

int main(int argc, char** argv) {
    try {
#ifdef ARB_MPI_ENABLED
        arbenv::with_mpi guard(argc, argv, false);
#endif
        auto opt = parse_options(argc, argv);
        if (!opt) return 0;

        arb::context ctx;
        if (opt->dry_run_ranks) {
            ctx = arb::make_context({}, arb::dry_run_info(opt->dry_run_ranks, opt->cells_per_rank));
        }
        else {
#ifdef ARB_MPI_ENABLED
            ctx = arb::make_context({}, MPI_COMM_WORLD);
#else
            ctx = arb::make_context();
#endif
        }
        unsigned rank = arb::rank(ctx);
        unsigned num_ranks = arb::num_ranks(ctx);
        std::cout << sup::mask_stream(rank == 0);

        auto rec = arb::symmetric_recipe(std::make_unique<ring_tile>(*opt, num_ranks));
        // In a dry run, rank 0's neighbourhoods are shifted to every other rank.
        auto exchange = ring_neighborhoods(rank, num_ranks, opt->widths);

        std::cout << "ranks:        " << num_ranks << (opt->dry_run_ranks? " (dry run)": "") << "\n"
                  << "cells:        " << rec.num_cells() << "\n\n";
        // Epochs are half the shortest delay. Like arbor, leave some slack for
        // rounding, which costs one epoch if a delay is a multiple of the epoch.
        auto epoch = *std::min_element(opt->delays.begin(), opt->delays.end())/2;
        auto describe = [&](const std::string& name, std::size_t g, std::size_t num_ranks_in_group) {
            std::cout << name << num_ranks_in_group << (num_ranks_in_group == 1? " rank": " ranks");
            if (!num_ranks_in_group) {
                std::cout << ", unused\n";
                return;
            }
            std::cout << ", " << opt->in_degrees[g] << " connections per cell, delay " << opt->delays[g] << " ms, ";
            if (g == 0) {
                std::cout << "no communication\n";
                return;
            }
            auto period = std::max(1, int(std::floor(opt->delays[g]/epoch*(1 - 1e-6))) - 1);
            std::cout << "exchanged every " << period << (period == 1? " epoch\n": " epochs\n");
        };
        describe("own rank:     ", 0, 1);
        std::size_t num_near = 1;
        for (std::size_t l = 0; l < opt->widths.size(); ++l) {
            describe("level " + std::to_string(l + 1) + ":      ", l + 1, exchange.neighborhoods[l].size());
            num_near += exchange.neighborhoods[l].size();
        }
        describe("far:          ", opt->widths.size() + 1, num_ranks - num_near);
        std::cout << "\n";

        auto global = simulate(rec, ctx, {}, *opt);
        auto neighborhood = simulate(rec, ctx, exchange, *opt);

        auto report = [](const std::string& name, const result& r) {
            std::cout << name << r.num_spikes << " spikes, init " << r.init_time << " s, run " << r.run_time << " s\n";
        };
        report("all-to-all:   ", global);
        report("neighborhood: ", neighborhood);
        if (neighborhood.num_spikes != global.num_spikes) {
            std::cerr << "error: the exchange methods produced different numbers of spikes\n";
            return 1;
        }
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in neighborhood example:\n" << e.what() << "\n";
        return 1;
    }
    return 0;
}

// Parse a comma separated list, e.g. "1,4".
template <typename T>
std::vector<T> parse_list(const std::string& str) {
    std::vector<T> values;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        T value;
        std::stringstream is(item);
        if (!(is >> value) || !is.eof()) throw std::runtime_error("invalid list: " + str);
        values.push_back(value);
    }
    return values;
}

template <typename T>
std::string to_list(const std::vector<T>& values) {
    std::stringstream ss;
    for (std::size_t i = 0; i < values.size(); ++i) ss << (i? ",": "") << values[i];
    return ss.str();
}

std::optional<options> parse_options(int argc, char** argv) {
    auto usage_str = "\n"
                     "-n|--cells-per-rank [Number of cells per rank]\n"
                     "-w|--widths         [Outer ring distance of each level, increasing, e.g. 1,4]\n"
                     "-i|--in-degrees     [Connections per cell from the own rank, each level, and far ranks]\n"
                     "-d|--delays         [Delays (ms) of these connections, e.g. 1,2,5,20]\n"
                     "-t|--tfinal         [Length of the simulation period (ms)]\n"
                     "-s|--dt             [Simulation time step (ms)]\n"
                     "-D|--dry-run        [Emulate this many ranks]\n";

    options opt;
    auto widths = to_list(opt.widths);
    auto in_degrees = to_list(opt.in_degrees);
    auto delays = to_list(opt.delays);
    auto help = [argv0 = argv[0], &usage_str] { to::usage(argv0, usage_str); };
    to::option cli[] = {
        { opt.cells_per_rank, "-n", "--cells-per-rank" },
        { widths,             "-w", "--widths" },
        { in_degrees,         "-i", "--in-degrees" },
        { delays,             "-d", "--delays" },
        { opt.tfinal,         "-t", "--tfinal" },
        { opt.dt,             "-s", "--dt" },
        { opt.dry_run_ranks,  "-D", "--dry-run" },
        { to::action(help), to::flag, to::exit, "-h", "--help" }
    };
    if (!to::run(cli, argc, argv + 1)) return {};
    if (argv[1]) throw to::option_error("unrecognized argument", argv[1]);

    opt.widths = parse_list<unsigned>(widths);
    opt.in_degrees = parse_list<unsigned>(in_degrees);
    opt.delays = parse_list<double>(delays);
    if (!std::is_sorted(opt.widths.begin(), opt.widths.end()) ||
        std::adjacent_find(opt.widths.begin(), opt.widths.end()) != opt.widths.end() ||
        (!opt.widths.empty() && opt.widths.front() == 0)) {
        throw std::runtime_error("widths must be positive and strictly increasing");
    }
    auto num_groups = opt.widths.size() + 2;
    if (opt.in_degrees.size() != num_groups || opt.delays.size() != num_groups) {
        throw std::runtime_error("need " + std::to_string(num_groups) +
                                 " in-degrees and delays: own rank, each level, and far ranks");
    }
    return opt;
}
