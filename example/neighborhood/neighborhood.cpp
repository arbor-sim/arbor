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
 * The model is simulated three times: with the default exchange, which runs a
 * global all-to-all every epoch (half the shortest delay), and twice with
 * neighbourhood exchange. There, each level is exchanged with a sparse MPI
 * neighbourhood all-gather and the far ranks with a global all-to-all, each
 * only as often as the delays from that group require. The levels are either
 * the ones described above, or chosen by arbor from the connection delays.
 * For timings, run each of these in its own process with --mode, so that they
 * do not influence each other.
 *
 * With --random-delays, each connection instead gets one of the --delays at
 * random, independent of the distance: the unfavourable case for neighbourhoods.
 *
 * Run with MPI, e.g. `mpirun -n 16 ./neighborhood --widths 1,4`, or emulate
 * many ranks with `./neighborhood --dry-run 64`.
 */

#include <algorithm>
#include <chrono>
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
#include <arbor/profile/profiler.hpp>
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
    double profile_threshold = 1;  // [%] hide profiler regions below this share of the wall time
    bool random_delays = false;    // delays independent of the distance
    std::string mode = "all";      // all-to-all, given, automatic, or all of them
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
        std::uniform_int_distribution<std::size_t> any_delay(0, opt_.delays.size() - 1);
        auto connect = [&](unsigned rank, double delay) {
            if (opt_.random_delays) delay = opt_.delays[any_delay(rng)];
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
    // Levels of the neighbourhood exchange on this rank, and the epoch length [ms].
    std::vector<arb::spike_exchange_level> levels;
    double epoch;
    // Profiler table of this rank; empty without profiling support.
    std::string profile;
};

result simulate(const arb::recipe& rec, arb::context ctx, const std::optional<arb::neighborhood_spike_exchange>& exchange, const options& opt) {
    using clock = std::chrono::steady_clock;
    auto seconds = [](clock::duration d) { return std::chrono::duration<double>(d).count(); };

#ifdef ARB_PROFILE_ENABLED
    // Profile this simulation only, from construction to the end of the run.
    arb::profile::profiler_clear();
#endif
    auto t0 = clock::now();
    auto sim = arb::simulation::create(rec).set_context(ctx);
    if (exchange) sim.set_spike_exchange(*exchange);
    auto simulation = sim.make_unique();
    auto t1 = clock::now();
    simulation->run(opt.tfinal*U::ms, opt.dt*U::ms);
    auto t2 = clock::now();

    std::stringstream profile;
#ifdef ARB_PROFILE_ENABLED
    arb::profile::print_profiler_summary(profile, opt.profile_threshold);
#endif
    return {simulation->num_spikes(), seconds(t1 - t0), seconds(t2 - t1),
            simulation->spike_exchange_levels(), simulation->max_epoch_length(), profile.str()};
}

// Print the levels arbor uses on this rank.
void print_levels(const std::string& title, const result& r) {
    std::cout << title << ", in epochs of " << r.epoch << " ms:\n";
    for (std::size_t l = 0; l < r.levels.size(); ++l) {
        const auto& level = r.levels[l];
        auto name = level.far? std::string("far"): "level " + std::to_string(l + 1);
        std::cout << "  " << name << ":" << std::string(10 - name.size(), ' ')
                  << level.ranks.size() << (level.ranks.size() == 1? " rank ": " ranks");
        // Long lists of ranks are cut short.
        if (!level.ranks.empty()) {
            std::cout << " [";
            for (std::size_t i = 0; i < level.ranks.size() && i < 8; ++i) std::cout << (i? ", ": "") << level.ranks[i];
            std::cout << (level.ranks.size() > 8? ", ...]": "]");
        }
        if (level.period) {
            std::cout << ", every " << level.period << (level.period == 1? " epoch": " epochs")
                      << " (" << level.period*r.epoch << " ms)\n";
        }
        else {
            std::cout << ", unused\n";
        }
    }
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
#ifdef ARB_PROFILE_ENABLED
        arb::profile::profiler_initialize(ctx);
#endif

        auto rec = arb::symmetric_recipe(std::make_unique<ring_tile>(*opt, num_ranks));
        // In a dry run, rank 0's neighbourhoods are shifted to every other rank.
        auto exchange = ring_neighborhoods(rank, num_ranks, opt->widths);

        std::cout << "ranks:        " << num_ranks << (opt->dry_run_ranks? " (dry run)": "") << "\n"
                  << "cells:        " << rec.num_cells() << "\n\n";

        // The model as seen from rank 0: groups of ranks with their connections.
        auto describe = [&](const std::string& name, std::size_t g, std::size_t num_ranks_in_group) {
            std::cout << "  " << name << num_ranks_in_group << (num_ranks_in_group == 1? " rank": " ranks");
            if (num_ranks_in_group) {
                std::cout << ", " << opt->in_degrees[g] << " connections per cell, delay " << opt->delays[g] << " ms";
            }
            std::cout << "\n";
        };
        std::cout << "model" << (opt->random_delays? ", with random delays from the list below": "") << ":\n";
        describe("own rank:  ", 0, 1);
        std::size_t num_near = 1;
        for (std::size_t l = 0; l < opt->widths.size(); ++l) {
            describe("level " + std::to_string(l + 1) + ":   ", l + 1, exchange.neighborhoods[l].size());
            num_near += exchange.neighborhoods[l].size();
        }
        describe("far:       ", opt->widths.size() + 1, num_ranks - num_near);
        std::cout << "\n";

        // Init includes making the neighbourhoods, for given and automatic levels alike.
        std::vector<std::pair<std::string, result>> results;
        auto run_mode = [&](const std::string& mode, const std::optional<arb::neighborhood_spike_exchange>& exchange) {
            if (opt->mode == "all" || opt->mode == mode) results.emplace_back(mode, simulate(rec, ctx, exchange, *opt));
        };
        run_mode("all-to-all", std::nullopt);
        run_mode("given", exchange);
        run_mode("automatic", arb::neighborhood_spike_exchange{.automatic = true});

        for (const auto& [name, r]: results) {
            if (name != "all-to-all") print_levels(name + " levels of rank 0", r);
        }
        std::cout << "\n";
        for (const auto& [name, r]: results) {
            std::cout << name << ":" << std::string(12 - name.size(), ' ') << r.num_spikes
                      << " spikes, init " << r.init_time << " s, run " << r.run_time << " s\n";
        }
        for (const auto& [name, r]: results) {
            if (r.num_spikes != results.front().second.num_spikes) {
                std::cerr << "error: the exchange methods produced different numbers of spikes\n";
                return 1;
            }
        }

#ifdef ARB_PROFILE_ENABLED
        for (const auto& [name, r]: results) {
            std::cout << "\nprofile of rank 0, " << name << ", regions above " << opt->profile_threshold << "%:\n"
                      << r.profile;
        }
#else
        std::cout << "\nfor profiler times, build arbor with -DARB_WITH_PROFILING=ON\n";
#endif
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
                     "-D|--dry-run        [Emulate this many ranks]\n"
                     "-P|--profile-threshold [Hide profiler regions below this share of the wall time (%)]\n"
                     "-r|--random-delays  [Give each connection one of the delays at random]\n"
                     "-m|--mode           [Exchange to run: all-to-all, given, automatic, or all]\n";

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
        { opt.profile_threshold, "-P", "--profile-threshold" },
        { to::set(opt.random_delays), to::flag, "-r", "--random-delays" },
        { opt.mode,           "-m", "--mode" },
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
    if (opt.mode != "all" && opt.mode != "all-to-all" && opt.mode != "given" && opt.mode != "automatic") {
        throw std::runtime_error("mode must be all-to-all, given, automatic, or all");
    }
    auto num_groups = opt.widths.size() + 2;
    if (opt.in_degrees.size() != num_groups || opt.delays.size() != num_groups) {
        throw std::runtime_error("need " + std::to_string(num_groups) +
                                 " in-degrees and delays: own rank, each level, and far ranks");
    }
    return opt;
}
