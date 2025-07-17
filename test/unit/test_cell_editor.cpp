#include <gtest/gtest.h>

#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/units.hpp>

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/benchmark_cell.hpp>
#include <arbor/spike_source_cell.hpp>

#include <arborenv/default_env.hpp>

#include "util/span.hpp"

constexpr double epsilon  = 1e-6;
#ifdef ARB_GPU_ENABLED
constexpr int    with_gpu = 0;
#else
constexpr int    with_gpu = -1;
#endif


using namespace arb::units::literals;

struct lif_recipe: arb::recipe {

    struct param_t {
        double weight = 0;
        double cm_pF = 0;
        size_t n_100 = 0;
        size_t n_200 = 0;
    };

    lif_recipe(double w, double cm_pf): weight(w), C_m(cm_pf*arb::units::pF) {}

    arb::cell_size_type num_cells() const override { return N; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::lif; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        auto cell = arb::lif_cell{"src", "tgt"};
        cell.C_m = C_m;
        return cell;
    }

    std::vector<arb::event_generator> event_generators(arb::cell_gid_type gid) const override {
        return {arb::regular_generator({"tgt"}, weight, 0_ms, 0.5_ms)};
    }

    arb::cell_size_type N = 10;

    double weight = 100;
    arb::units::quantity C_m = 20_pF;
};

TEST(edit_lif, no_edit) {
    using  param_t = lif_recipe::param_t;
    // check base case at 20pF
    //                               weight  c_m  t=100 200
    for (const auto& param: {param_t{  10.0, 20.0,  20,   50},
                             param_t{ 100.0, 20.0, 330,  670},
                             param_t{1000.0, 20.0, 500, 1000}}) {
        auto rec = lif_recipe{param.weight, param.cm_pF};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }
    // check base case at 40pF
    //                               weight  c_m  t=100 200
    for (const auto& param: {param_t{  10.0, 40.0,  00,    0},
                             param_t{ 100.0, 40.0, 250,  500},
                             param_t{1000.0, 40.0, 500, 1000}}) {
        auto rec = lif_recipe{param.weight, param.cm_pF};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }
}

TEST(edit_lif, edit) {
    using  param_t = lif_recipe::param_t;

    arb::lif_cell_editor edit = [](arb::lif_cell& cell) { cell.C_m = 40_pF; };

    auto ctx = arb::make_context();
    // scan group sizes
    for (auto group: arb::util::make_span(1, 10)) {
        auto phm = arb::partition_hint_map{
            {arb::cell_kind::lif,
             arb::partition_hint{
                 .cpu_group_size=std::size_t(group),
                 .gpu_group_size=std::size_t(group),
                 .prefer_gpu=true,
             }
             }
        };
        // check transition from 20pF -> 40pF for cell gid=0
        //                               weight  c_m   t=100 200
        for (const auto& param: {param_t{  10.0, 20.0,  20,   47},
                                 param_t{ 100.0, 20.0, 330,  661},
                                 param_t{1000.0, 20.0, 500, 1000}}) {
            auto rec = lif_recipe{param.weight, param.cm_pF};
            auto ddc = arb::partition_load_balance(rec, ctx, phm);
            auto sim = arb::simulation{rec, ctx, ddc};
            sim.run(100_ms, 0.1_ms);
            EXPECT_EQ(sim.num_spikes(), param.n_100);
            sim.edit_cell(0, edit);
            sim.run(200_ms, 0.1_ms);
            EXPECT_EQ(sim.num_spikes(), param.n_200);
        }
        // check transition from 20pF -> 40pF for half of cells
        //                               weight  c_m   t=100 200
        for (const auto& param: {param_t{  10.0, 20.0,  20,   35},
                                 param_t{ 100.0, 20.0, 330,  625},
                                 param_t{1000.0, 20.0, 500, 1000}}) {
            auto rec = lif_recipe{param.weight, param.cm_pF};
            auto ddc = arb::partition_load_balance(rec, ctx, phm);
            auto sim = arb::simulation{rec, ctx, ddc};
            sim.run(100_ms, 0.1_ms);
            EXPECT_EQ(sim.num_spikes(), param.n_100);
            for (arb::cell_gid_type gid = 0; gid < rec.num_cells(); gid += 2) sim.edit_cell(gid, edit);
            sim.run(200_ms, 0.1_ms);
            EXPECT_EQ(sim.num_spikes(), param.n_200);
        }
        // check transition from 20pF -> 40pF for all cells
        //                               weight  c_m   t=100 200
        for (const auto& param: {param_t{  10.0, 20.0,  20,   20},
                                 param_t{ 100.0, 20.0, 330,  580},
                                 param_t{1000.0, 20.0, 500, 1000}}) {
            auto rec = lif_recipe{param.weight, param.cm_pF};
            auto ddc = arb::partition_load_balance(rec, ctx, phm);
            auto sim = arb::simulation{rec, ctx, ddc};
            sim.run(100_ms, 0.1_ms);
            EXPECT_EQ(sim.num_spikes(), param.n_100);
            for (arb::cell_gid_type gid = 0; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
            sim.run(200_ms, 0.1_ms);
            EXPECT_EQ(sim.num_spikes(), param.n_200);
        }
        // edits are idempotent
        // check transition from 20pF -> 40pF for all cells
        //                               weight  c_m   t=100 200
        for (const auto& param: {param_t{  10.0, 20.0,  20,   20},
                                 param_t{ 100.0, 20.0, 330,  580},
                                 param_t{1000.0, 20.0, 500, 1000}}) {
            auto rec = lif_recipe{param.weight, param.cm_pF};
            auto ddc = arb::partition_load_balance(rec, ctx, phm);
            auto sim = arb::simulation{rec, ctx, ddc};
            sim.run(100_ms, 0.1_ms);
            EXPECT_EQ(sim.num_spikes(), param.n_100);
            for (arb::cell_gid_type gid = 0; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
            for (arb::cell_gid_type gid = 0; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
            sim.run(200_ms, 0.1_ms);
            EXPECT_EQ(sim.num_spikes(), param.n_200);
        }

    }
}

TEST(edit_lif, errors) {
    auto rec = lif_recipe{0, 0};
    auto sim = arb::simulation{rec};
    // Check that errors are actually thrown.
    EXPECT_THROW(sim.edit_cell( 0, arb::lif_cell_editor([](auto& cell) { cell.V_m = 42_mV; })), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell( 0, arb::lif_cell_editor([](auto& cell) { cell.source = "foo"; })), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell( 0, arb::lif_cell_editor([](auto& cell) { cell.target = "foo"; })), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell( 0, 42), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell(42, arb::lif_cell_editor([](arb::lif_cell& cell) { cell.C_m = 40_pF; })), std::range_error);
}

struct bench_recipe: arb::recipe {

    struct param_t {
        double rtr = 1.0;    // real time
        double nu_kHz = 1.0; // freq
        size_t n_100 = 0;    // spikes after N ms
        size_t n_200 = 0;
    };

    bench_recipe(double r, double f): ratio(r), freq(f*arb::units::kHz) {}

    arb::cell_size_type num_cells() const override { return N; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::benchmark; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        return arb::benchmark_cell{.source="src", .target="tgt", .time_sequence=arb::regular_schedule(1/freq), .realtime_ratio=ratio};
    }

    arb::cell_size_type N = 10;

    double ratio = 1.0;
    arb::units::quantity freq = 1_kHz;
};

TEST(edit_bench, no_edit) {
    using param_t = bench_recipe::param_t;
    //                               rtr  nu   100  200
    for (const auto& param: {param_t{1e-4, 1.0, 1000, 2000}, // 10 cells x 100ms x 1kHz = 1000 spikes
                             param_t{1e-4, 2.0, 2000, 4000},
                             param_t{1e-4, 4.0, 4000, 8000}}) {
        auto rec = bench_recipe{param.rtr, param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }
}

TEST(edit_bench, edit_rate) {
    using param_t = bench_recipe::param_t;
    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1e-4, 1.0, 1000, 2100}, // one cell adds 100ms x 2kHz, the others stay at 1Khz => 100 spikes extra
                             param_t{1e-4, 2.0, 2000, 4200},
                             param_t{1e-4, 4.0, 4000, 8400}}) {
        arb::benchmark_cell_editor edit = [&](auto& cell) { cell.time_sequence = arb::regular_schedule(1.0/(2.0_kHz * param.nu_kHz)); };
        auto rec = bench_recipe{param.rtr, param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        sim.edit_cell(5, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }

    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1e-4, 1.0, 1000,  2500}, // one cell adds 100ms x 2kHz, the others stay at 1Khz => 100 spikes extra
                             param_t{1e-4, 2.0, 2000,  5000},
                             param_t{1e-4, 4.0, 4000, 10000}}) {
        arb::benchmark_cell_editor edit = [&](auto& cell) { cell.time_sequence = arb::regular_schedule(1.0/(2.0_kHz * param.nu_kHz)); };
        auto rec = bench_recipe{param.rtr, param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        for (auto gid = 0u; gid < rec.num_cells(); gid += 2) sim.edit_cell(gid, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }

    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1e-4, 1.0, 1000,  3000}, // one cell adds 100ms x 2kHz, the others stay at 1Khz => 100 spikes extra
                             param_t{1e-4, 2.0, 2000,  6000},
                             param_t{1e-4, 4.0, 4000, 12000}}) {
        arb::benchmark_cell_editor edit = [&](auto& cell) { cell.time_sequence = arb::regular_schedule(1.0/(2.0_kHz * param.nu_kHz)); };
        auto rec = bench_recipe{param.rtr, param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        for (auto gid = 0u; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }

}

TEST(edit_bench, edit_schedule) {
    auto eps = 0.06;

    using param_t = bench_recipe::param_t;
    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1e-4, 1.0, 1000, 2000}, // one cell adds 100ms x 2kHz, the others stay at 1Khz => 100 spikes extra
                             param_t{1e-4, 2.0, 2000, 4000},
                             param_t{1e-4, 4.0, 4000, 8000}}) {
        arb::benchmark_cell_editor edit = [&](auto& cell) { cell.time_sequence = arb::poisson_schedule(1.0_ms/param.nu_kHz); };
        auto rec = bench_recipe{param.rtr, param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        sim.edit_cell(5, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_GE(sim.num_spikes(), param.n_200*(1.0 - eps));
        EXPECT_LE(sim.num_spikes(), param.n_200*(1.0 + eps));

    }

    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1e-4, 1.0, 1000, 2000}, // one cell adds 100ms x 2kHz, the others stay at 1Khz => 100 spikes extra
                             param_t{1e-4, 2.0, 2000, 4000},
                             param_t{1e-4, 4.0, 4000, 8000}}) {
        arb::benchmark_cell_editor edit = [&](auto& cell) { cell.time_sequence = arb::poisson_schedule(1.0_ms/param.nu_kHz); };
        auto rec = bench_recipe{param.rtr, param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        for (auto gid = 0u; gid < rec.num_cells(); gid += 2) sim.edit_cell(gid, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_GE(sim.num_spikes(), param.n_200*(1.0 - eps));
        EXPECT_LE(sim.num_spikes(), param.n_200*(1.0 + eps));
    }

    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1e-4, 1.0, 1000, 2000}, // one cell adds 100ms x 2kHz, the others stay at 1Khz => 100 spikes extra
                             param_t{1e-4, 2.0, 2000, 4000},
                             param_t{1e-4, 4.0, 4000, 8000}}) {
        arb::benchmark_cell_editor edit = [&](auto& cell) { cell.time_sequence = arb::poisson_schedule(1.0_ms/param.nu_kHz); };
        auto rec = bench_recipe{param.rtr, param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        for (auto gid = 0u; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_GE(sim.num_spikes(), param.n_200*(1.0 - eps));
        EXPECT_LE(sim.num_spikes(), param.n_200*(1.0 + eps));
    }
}

TEST(edit_benchmark, errors) {
    auto rec = bench_recipe{1, 1};
    auto sim = arb::simulation{rec};
    // Check that errors are actually thrown.
    EXPECT_THROW(sim.edit_cell( 0, arb::benchmark_cell_editor([](auto& cell) { cell.source = "foo"; })), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell( 0, arb::benchmark_cell_editor([](auto& cell) { cell.target = "foo"; })), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell( 0, 42), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell(42, arb::benchmark_cell_editor([](auto& cell) { cell.realtime_ratio = 42; })), std::range_error);
}

TEST(edit_benchmark, do_nothing_does_nothing) {
    arb::benchmark_cell_editor edit = [](auto& cell) { cell.time_sequence = arb::poisson_schedule(1.0_kHz, 42);};
    arb::benchmark_cell_editor noop = [](auto& cell) {};

    size_t n_noop = 0;
    {
        auto rec = bench_recipe{1e-4, 1.0};
        auto sim = arb::simulation{rec};
        for (auto gid = 0u; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
        sim.run(100_ms, 0.1_ms);
        for (auto gid = 0u; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, noop);
        sim.run(200_ms, 0.1_ms);
        n_noop = sim.num_spikes();
    }
    size_t n_expt = 0;
    {
        auto rec = bench_recipe{1e-4, 1.0};
        auto sim = arb::simulation{rec};
        for (auto gid = 0u; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
        sim.run(200_ms, 0.1_ms);
        n_expt = sim.num_spikes();
    }
    EXPECT_EQ(n_expt, n_noop);
    EXPECT_GE(n_noop, 2100);
    EXPECT_LE(n_noop, 2400);
}


struct source_recipe: arb::recipe {

    struct param_t {
        double nu_kHz = 1.0; // freq
        size_t n_100 = 0;    // spikes after N ms
        size_t n_200 = 0;
    };

    source_recipe(double f): freq(f*arb::units::kHz) {}

    arb::cell_size_type num_cells() const override { return N; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::spike_source; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        return arb::spike_source_cell{"tgt", arb::regular_schedule(1/freq)};
    }

    arb::cell_size_type N = 10;
    arb::units::quantity freq = 1_kHz;
};

TEST(edit_source, no_edit) {
    using param_t = source_recipe::param_t;
    //                               rtr  nu   100  200
    for (const auto& param: {param_t{1.0, 1000, 2000}, // 10 cells x 100ms x 1kHz = 1000 spikes
                             param_t{2.0, 2000, 4000},
                             param_t{4.0, 4000, 8000}}) {
        auto rec = source_recipe{param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }
}

TEST(edit_source, edit_rate) {
    using param_t = source_recipe::param_t;
    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1.0, 1000, 2100}, // one cell adds 100ms x 2kHz, the others stay at 1Khz => 100 spikes extra
                             param_t{2.0, 2000, 4200},
                             param_t{4.0, 4000, 8400}}) {
        arb::spike_source_cell_editor edit = [&](auto& cell) { cell.schedules = {arb::regular_schedule(1.0/(2.0_kHz * param.nu_kHz))}; };
        auto rec = source_recipe{param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        sim.edit_cell(5, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }

    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1.0, 1000,  2500}, // one cell adds 100ms x 2kHz, the others stay at 1Khz => 100 spikes extra
                             param_t{2.0, 2000,  5000},
                             param_t{4.0, 4000, 10000}}) {
        arb::spike_source_cell_editor edit = [&](auto& cell) { cell.schedules = {arb::regular_schedule(1.0/(2.0_kHz * param.nu_kHz))}; };
        auto rec = source_recipe{param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        for (auto gid = 0u; gid < rec.num_cells(); gid += 2) sim.edit_cell(gid, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }

    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1.0, 1000,  3000}, // one cell adds 100ms x 2kHz, the others stay at 1Khz => 100 spikes extra
                             param_t{2.0, 2000,  6000},
                             param_t{4.0, 4000, 12000}}) {
        arb::spike_source_cell_editor edit = [&](auto& cell) { cell.schedules = {arb::regular_schedule(1.0/(2.0_kHz * param.nu_kHz))}; };
        auto rec = source_recipe{param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        for (auto gid = 0u; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }

}

TEST(edit_source, do_nothing_does_nothing) {
    arb::spike_source_cell_editor edit = [](auto& cell) { cell.schedules = {arb::poisson_schedule(1.0_kHz, 42) };};
    arb::spike_source_cell_editor noop = [](arb::spike_source_cell& cell) {};

    size_t n_noop = 0;
    {
        auto rec = source_recipe{1.0};
        auto sim = arb::simulation{rec};
        for (auto gid = 0u; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
        sim.run(100_ms, 0.1_ms);
        for (auto gid = 0u; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, noop);
        sim.run(200_ms, 0.1_ms);
        n_noop = sim.num_spikes();
    }
    size_t n_expt = 0;
    {
        auto rec = source_recipe{1.0};
        auto sim = arb::simulation{rec};
        for (auto gid = 0u; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
        sim.run(200_ms, 0.1_ms);
        n_expt = sim.num_spikes();
    }
    EXPECT_EQ(n_expt, n_noop);
    EXPECT_GE(n_noop, 2100);
    EXPECT_LE(n_noop, 2400);
}

TEST(edit_source, edit_schedule) {
    auto eps = 0.06;

    using param_t = source_recipe::param_t;
    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1.0, 1000, 2100}, // one cell adds 100ms x 2kHz, the others stay at 1Khz => 100 spikes extra
                             param_t{2.0, 2000, 4100},
                             param_t{4.0, 4000, 8100}}) {
        arb::spike_source_cell_editor edit = [&](auto& cell) { cell.schedules.push_back(arb::poisson_schedule(1.0_ms/param.nu_kHz)); };
        auto rec = source_recipe{param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        sim.edit_cell(5, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_GE(sim.num_spikes(), param.n_200*(1.0 - eps));
        EXPECT_LE(sim.num_spikes(), param.n_200*(1.0 + eps));
    }

    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1.0, 1000,  2500},
                             param_t{2.0, 2000,  5000},
                             param_t{4.0, 4000, 10000}}) {
        arb::spike_source_cell_editor edit = [&](auto& cell) { cell.schedules.push_back(arb::poisson_schedule(1.0_ms/param.nu_kHz)); };
        auto rec = source_recipe{param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        for (auto gid = 0u; gid < rec.num_cells(); gid += 2) sim.edit_cell(gid, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_GE(sim.num_spikes(), param.n_200*(1.0 - eps));
        EXPECT_LE(sim.num_spikes(), param.n_200*(1.0 + eps));
    }

    //                               rtr  nu    100   200
    for (const auto& param: {param_t{1.0, 1000,  3000},  // 0-100: 1000 spikes 100-200: 1000 + ~1000
                             param_t{2.0, 2000,  6000},
                             param_t{4.0, 4000, 12000}}) {
        arb::spike_source_cell_editor edit = [&](auto& cell) { cell.schedules.push_back(arb::poisson_schedule(1.0_ms/param.nu_kHz)); };
        auto rec = source_recipe{param.nu_kHz};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        for (auto gid = 0u; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_GE(sim.num_spikes(), param.n_200*(1.0 - eps));
        EXPECT_LE(sim.num_spikes(), param.n_200*(1.0 + eps));
    }
}

TEST(edit_spike_source, errors) {
    auto rec = source_recipe{1};
    auto sim = arb::simulation{rec};
    // Check that errors are actually thrown.
    EXPECT_THROW(sim.edit_cell( 0, arb::spike_source_cell_editor([](auto& cell) { cell.source = "foo"; })), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell( 0, 42), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell(42, arb::spike_source_cell_editor([](auto& cell) {})), std::range_error);
}

constexpr size_t N = 4;
constexpr double eps = 1e-6;
constexpr double T = 40;
constexpr double dt = 1;
constexpr size_t n_step = T/dt;
using result_t = std::vector<std::array<double, N>>;

struct cable_recipe: arb::recipe {
    cable_recipe() {
        props.default_parameters = arb::neuron_parameter_defaults;
    }

    arb::cell_size_type num_cells() const override { return N; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::cable; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        // Create a cable cell
        //
        //   +------+
        //   |  hh  |=== pas ===
        //   +------+
        //
        auto dec = arb::decor{}
            .paint(arb::reg::tagged(1), arb::density("hh", {{"gkbar", 0.036}}))
            .paint(arb::reg::tagged(2), arb::density("pas"))
            .place(arb::ls::location(0, 0.5), arb::i_clamp::box(10_ms, 20_ms, 100_pA), "ic1")
            ;
        auto par = arb::mnpos;
        auto seg = arb::segment_tree{};
        par = seg.append(par, { 0, 0, 0, 42}, {10, 0, 0, 42}, 1); // soma
        par = seg.append(par, {10, 0, 0, 23}, {20, 0, 0, 23}, 2); // dendrite
        auto mrf = arb::morphology{seg};
        auto lbl = arb::label_dict{};
        auto cvp = arb::cv_policy_max_extent(1.0);
        return arb::cable_cell{mrf, dec, lbl, cvp};
    }

    virtual std::vector<arb::probe_info> get_probes(arb::cell_gid_type gid) const override { return {{arb::cable_probe_membrane_voltage{arb::ls::location(0, 0.5)}, "Um"}}; }
    std::any get_global_properties(arb::cell_kind) const override { return props; }

    arb::cable_cell_global_properties props;

};

testing::AssertionResult all_near(const std::vector<double>& a, const result_t& b, int iy, double eps) {
    if (a.size() != b.size()) return testing::AssertionFailure() << "sequences differ in length"
                                                                 << " #expected=" << b.size()
                                                                 << " #received=" << a.size();
    std::stringstream res;
    res << std::setprecision(9);
    for (size_t ix = 0; ix < a.size(); ++ix) {
        // printf("%9.6f, ", b[ix][iy]);
        auto ax = a[ix];
        auto bx = b[ix][iy];
        if (fabs(ax - bx) > eps) {
            res << " elements " << ax << " and " << bx << " differ at index " << ix << ", " << iy << ".";
            break;
        }
    }
    std::string str = res.str();
    std::cerr << res.str();
    if (str.empty()) return testing::AssertionSuccess();
    else             return testing::AssertionFailure() << str;
}

TEST(edit_cable, errors) {
    auto rec = cable_recipe{};
    auto sim = arb::simulation{rec};
    // wrong editor
    EXPECT_THROW(sim.edit_cell( 0, arb::spike_source_cell_editor([](auto& cell) { cell.source = "foo"; })), arb::bad_cell_edit);
    // non-existant gid
    EXPECT_THROW(sim.edit_cell(42, arb::cable_cell_density_editor{.mechanism="hh",  .values={{"gnabar", 23}}}), std::range_error);
    // non-existant mechanism
    EXPECT_THROW(sim.edit_cell( 0, arb::cable_cell_density_editor{.mechanism="bar", .values={}}), arb::bad_cell_edit);
    // non-existant parameter
    EXPECT_THROW(sim.edit_cell( 0, arb::cable_cell_density_editor{.mechanism="hh",  .values={{"foobar", 23}}}), arb::bad_cell_edit);
}

TEST(edit_cable, hh) {
    result_t sample_values;
    sample_values.resize(n_step);
    auto sampler = [&sample_values](arb::probe_metadata pm, std::size_t n, const arb::sample_record* samples) {
        auto gid = pm.id.gid;
        for (std::size_t ix = 0; ix < n; ++ix) {
            sample_values[ix][gid] = *arb::util::any_cast<const double*>(samples[ix].data);
        }
    };

    std::vector unedited = {-65.000000, -65.976650, -66.650927, -67.003375, -67.167843, -67.211650, -67.190473, -67.136324, -67.069902, -67.002777, -66.941313, -65.466084, -64.416894,  -63.769651, -63.388147, -63.232318, -63.250644, -63.392280, -63.602123, -63.830035, -64.038928, -64.208130, -64.331221, -64.411239, -64.455972, -64.474584, -64.475653, -64.466276, -64.451815, -64.436002, -64.421192, -65.752674, -66.673119, -67.139232, -67.345711, -67.391215, -67.353235, -67.274783, -67.182795, -67.091783};
    std::vector   edited = {-65.000000, -67.510461, -68.850726, -69.323006, -69.412256, -69.310662, -69.143188, -68.961183, -68.793752, -68.650870, -68.535689, -67.105997, -66.245441, -65.838325, -65.701969, -65.744705, -65.876993, -66.038533, -66.190518, -66.314565, -66.405806, -66.467100, -66.504513, -66.524656, -66.533345, -66.535117, -66.533209, -66.529750, -66.526015, -66.522677, -66.520013, -67.778855, -68.485211, -68.769411, -68.846721, -68.811386, -68.730803, -68.636385, -68.546405, -68.468299 };

    auto ctx = arb::make_context({arbenv::default_concurrency(), with_gpu});
    auto rec = cable_recipe{};

    // results must be invariant under the group size, even if it doesn't divide into N
    for (size_t g_size = 1; g_size <= N; ++g_size) {
        // ... and the gid targeted
        for (size_t gid = 0; gid < N; ++gid) {
            auto sim = arb::simulation{rec, ctx, partition_load_balance(rec, ctx, {{arb::cell_kind::cable, arb::partition_hint{.cpu_group_size=g_size}}})};
            sim.add_sampler(arb::all_probes, arb::regular_schedule(dt*arb::units::ms), sampler);
            sim.edit_cell(gid, arb::cable_cell_density_editor{.mechanism="hh", .values={{"gkbar", 0.08}}});
            sim.run(T*arb::units::ms, dt*arb::units::ms);
            // all gids present 'unedited' traces, except the one gid we targeted
            for (size_t col = 0; col < N; ++col) {
                if (col == gid) {
                    EXPECT_TRUE(all_near(edited, sample_values, col, eps));
                }
                else {
                    EXPECT_TRUE(all_near(unedited, sample_values, col, eps));
                }
            }
        }
    }
}

TEST(edit_cable, pas) {
    result_t sample_values;
    sample_values.resize(n_step);
    auto sampler = [&sample_values](arb::probe_metadata pm, std::size_t n, const arb::sample_record* samples) {
        auto gid = pm.id.gid;
        for (std::size_t ix = 0; ix < n; ++ix) {
            sample_values[ix][gid] = *arb::util::any_cast<const double*>(samples[ix].data);
        }
    };

    std::vector unedited = {-65.000000, -65.976650, -66.650927, -67.003375, -67.167843, -67.211650, -67.190473, -67.136324, -67.069902, -67.002777, -66.941313, -65.466084, -64.416894,  -63.769651, -63.388147, -63.232318, -63.250644, -63.392280, -63.602123, -63.830035, -64.038928, -64.208130, -64.331221, -64.411239, -64.455972, -64.474584, -64.475653, -64.466276, -64.451815, -64.436002, -64.421192, -65.752674, -66.673119, -67.139232, -67.345711, -67.391215, -67.353235, -67.274783, -67.182795, -67.091783};
    std::vector   edited = {-65.000000, -69.757544, -69.936277, -69.930119, -69.928897, -69.923681, -69.921390, -69.918695, -69.916944, -69.915342, -69.914131, -69.830342, -69.826531, -69.825903, -69.825334, -69.824923, -69.824554, -69.824259, -69.824007, -69.823799, -69.823624, -69.823477, -69.823354, -69.823251, -69.823164, -69.823091, -69.823029, -69.822978, -69.822934, -69.822897, -69.822866, -69.905645, -69.908623, -69.908538, -69.908525, -69.908448, -69.908413, -69.908370, -69.908339, -69.908311 };

    auto ctx = arb::make_context({arbenv::default_concurrency(), with_gpu});
    auto rec = cable_recipe{};

    // results must be invariant under the group size, even if it doesn't divide into N
    for (size_t g_size = 1; g_size <= N; ++g_size) {
        // ... and the gid targeted
        for (size_t gid = 0; gid < N; ++gid) {
            auto sim = arb::simulation{rec, ctx, partition_load_balance(rec, ctx, {{arb::cell_kind::cable, arb::partition_hint{.cpu_group_size=g_size}}})};
            sim.add_sampler(arb::all_probes, arb::regular_schedule(dt*arb::units::ms), sampler);
            sim.edit_cell(gid, arb::cable_cell_density_editor{.mechanism="pas", .values={{"g", 0.08}}});
            sim.run(T*arb::units::ms, dt*arb::units::ms);
            // all gids present 'unedited' traces, except the one gid we targeted
            for (size_t col = 0; col < N; ++col) {
                if (col == gid) {
                    EXPECT_TRUE(all_near(edited, sample_values, col, eps));
                }
                else {
                    EXPECT_TRUE(all_near(unedited, sample_values, col, eps));
                }
            }
        }
    }
}
