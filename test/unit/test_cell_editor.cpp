#include <gtest/gtest.h>

#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/units.hpp>

#include <arbor/lif_cell.hpp>
#include <arbor/benchmark_cell.hpp>
#include <arbor/spike_source_cell.hpp>

#include "util/span.hpp"

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
    arb::spike_source_cell_editor noop = [](auto& cell) {};

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
