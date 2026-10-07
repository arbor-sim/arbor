#include <gtest/gtest.h>

#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/units.hpp>

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/adex_cell.hpp>
#include <arbor/benchmark_cell.hpp>
#include <arbor/spike_source_cell.hpp>

#include <arborenv/default_env.hpp>

#include "util/span.hpp"

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

struct adex_recipe: arb::recipe {

    struct param_t {
        double weight = 0;
        double cm_pF = 0;
        size_t n_100 = 0;
        size_t n_200 = 0;
    };

    adex_recipe(double w, double cm_pf): weight(w), C_m(cm_pf*arb::units::pF) {}

    arb::cell_size_type num_cells() const override { return N; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::adex; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        auto cell = arb::adex_cell{"src", "tgt"};
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

TEST(edit_adex, no_edit) {
    using  param_t = adex_recipe::param_t;
    // check base case at 20pF
    //                               weight  c_m  t=100 200
    for (const auto& param: {param_t{ 1.5, 20.0, 210, 300 },
                             param_t{ 2.5, 20.0, 360, 630 },
                             param_t{ 3.0, 20.0, 370, 710 }}) {
        auto rec = adex_recipe{param.weight, param.cm_pF};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }
    // check base case at 40pF
    //                               weight  c_m  t=100 200
    for (const auto& param: {param_t{ 1.5, 40.0, 130, 170 },
                             param_t{ 2.5, 40.0, 310, 520 },
                             param_t{ 3.0, 40.0, 340, 620 }}) {
        auto rec = adex_recipe{param.weight, param.cm_pF};
        auto sim = arb::simulation{rec};
        sim.run(100_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_100);
        sim.run(200_ms, 0.1_ms);
        EXPECT_EQ(sim.num_spikes(), param.n_200);
    }
}

TEST(edit_adex, edit) {
    using  param_t = adex_recipe::param_t;

    arb::adex_cell_editor edit = [](arb::adex_cell& cell) { cell.C_m = 40_pF; };

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
        //                               weight c_m   t=100 200
        for (const auto& param: {param_t{ 1.5,  20.0, 210,  304},
                                 param_t{ 2.5,  20.0, 360,  634},
                                 param_t{ 3.0,  20.0, 370,  709}}) {
            auto rec = adex_recipe{param.weight, param.cm_pF};
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
        for (const auto& param: {param_t{ 1.5,  20.0, 210,  320},
                                 param_t{ 2.5,  20.0, 360,  650},
                                 param_t{ 3.0,  20.0, 370,  705}}) {
            auto rec = adex_recipe{param.weight, param.cm_pF};
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
        for (const auto& param: {param_t{ 1.5,  20.0, 210,  340},
                                 param_t{ 2.5,  20.0, 360,  670},
                                 param_t{ 3.0,  20.0, 370,  700}}) {
            auto rec = adex_recipe{param.weight, param.cm_pF};
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
        for (const auto& param: {param_t{ 1.5,  20.0, 210,  340},
                                 param_t{ 2.5,  20.0, 360,  670},
                                 param_t{ 3.0,  20.0, 370,  700}}) {
            auto rec = adex_recipe{param.weight, param.cm_pF};
            auto ddc = arb::partition_load_balance(rec, ctx, phm);
            auto sim = arb::simulation{rec, ctx, ddc};
            sim.run(100_ms, 0.1_ms);
            EXPECT_EQ(sim.num_spikes(), param.n_100);
            for (arb::cell_gid_type gid = 0; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
            for (arb::cell_gid_type gid = 0; gid < rec.num_cells(); ++gid) sim.edit_cell(gid, edit);
            sim.run(200_ms, 0.1_ms);
            EXPECT_EQ(sim.num_spikes(), param.n_200);
        }
        break;
    }
}

TEST(edit_adex, errors) {
    auto rec = adex_recipe{0, 0};
    auto sim = arb::simulation{rec};
    // Check that errors are actually thrown.
    EXPECT_THROW(sim.edit_cell( 0, arb::adex_cell_editor([](auto& cell) { cell.V_m = 42_mV; })), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell( 0, arb::adex_cell_editor([](auto& cell) { cell.source = "foo"; })), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell( 0, arb::adex_cell_editor([](auto& cell) { cell.target = "foo"; })), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell( 0, 42), arb::bad_cell_edit);
    EXPECT_THROW(sim.edit_cell(42, arb::adex_cell_editor([](auto&& cell) { cell.C_m = 40_pF; })), std::range_error);
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
        arb::benchmark_cell_editor edit = [&](auto& cell) {
            cell.time_sequence = arb::regular_schedule(1.0/(2.0_kHz * param.nu_kHz));
        };
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
    // allow -- due to the stochastic nature -- up to
    auto eps = 0.15;

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
    EXPECT_GE(n_noop, 2000);
    EXPECT_LE(n_noop, 2100);
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
    EXPECT_GE(n_noop, 2000);
    EXPECT_LE(n_noop, 2100);
}

TEST(edit_source, edit_schedule) {
    auto eps = 0.15;

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
    for (const auto& param: {param_t{1.0, 1000,  3100},  // 0-100: 1000 spikes 100-200: 1000 + ~1000
                             param_t{2.0, 2000,  6100},
                             param_t{4.0, 4000, 12100}}) {
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
    cable_recipe(arb::decor dec_): dec{std::move(dec_)} {
        props.default_parameters = arb::neuron_parameter_defaults;
    }

    arb::cell_size_type num_cells() const override { return N; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::cable; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        // Create a cable cell
        //
        //   +------+
        //   |  hh  |=== pas ===
        //   +------+
        //
        auto par = arb::mnpos;
        auto seg = arb::segment_tree{};
        par = seg.append(par, { 0, 0, 0, 42}, {10, 0, 0, 42}, 1); // soma
        par = seg.append(par, {10, 0, 0, 23}, {20, 0, 0, 23}, 2); // dendrite
        auto mrf = arb::morphology{seg};
        auto lbl = arb::label_dict{};
        return arb::cable_cell{mrf, dec, lbl, cvp, arb::cable_cell_mutability::enabled};
    }

    virtual std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override { return {{arb::cable_probe_membrane_voltage{arb::ls::location(0, 0.5)}, "Um"}}; }
    std::any get_global_properties(arb::cell_kind) const override { return props; }

    arb::cable_cell_global_properties props;
    arb::decor dec;
    arb::cv_policy cvp = arb::cv_policy_max_extent(1.0_um);
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
            res << " elements " << ax << " and " << bx << " differ at index " << ix << ", " << iy;
            break;
        }
    }
    std::string str = res.str();
    std::cerr << res.str();
    if (str.empty()) return testing::AssertionSuccess();
    else             return testing::AssertionFailure() << str;
}

TEST(edit_cable, errors) {
    auto dec = arb::decor{}
        .paint(arb::reg::tagged(1), arb::density("hh", {{"gkbar", 0.036}}))
        .paint(arb::reg::tagged(2), arb::density("pas"))
        .place(arb::ls::location(0, 0.5), arb::i_clamp::box(10_ms, 20_ms, 100_pA), "ic1")
        ;

    auto rec = cable_recipe{dec};
    auto sim = arb::simulation{rec};
    // wrong editor
    EXPECT_THROW(sim.edit_cell( 0, arb::spike_source_cell_editor([](auto& cell) { cell.source = "foo"; })), arb::bad_cell_edit);
    // non-existant gid
    EXPECT_THROW(sim.edit_cell(42,
                               arb::cable_cell_editor {
                                   .on_density = [](const arb::region& where, const std::string& what, const arb::parameter_map& old) {
                                       return old;
                                   },
                               }),
                 std::range_error);
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

    std::vector unedited = {-65.000000, -65.976650, -66.650927,  -67.003375, -67.167843, -67.211650, -67.190473, -67.136324, -67.069902, -67.002777, -66.941313, -65.466084, -64.416894, -63.769651, -63.388147, -63.232318, -63.250644, -63.392280, -63.602123, -63.830035, -64.038928, -64.208130, -64.331221, -64.411239, -64.455972, -64.474584, -64.475653, -64.466276, -64.451815, -64.436002, -64.421192, -65.752674, -66.673119, -67.139232, -67.345711, -67.391215, -67.353235, -67.274783, -67.182795, -67.091783 };
    std::vector   edited = {-65.000000, -64.712831, -64.4678048, -64.284311, -64.1428594, -64.0383144, -63.9637155, -63.9138841, -63.8840799, -63.8701359, -63.8683377, -62.3633429, -60.8617346, -59.3328255, -57.403936, -54.2563308, -46.4880246, -13.1916514, 36.3749165, -2.63933723, -32.8848184, -51.0183638, -65.6607921, -69.4867711, -69.6836194, -68.922367, -67.7889208, -66.6558301, -65.5674181, -64.5892848, -63.701057, -64.349117, -64.8180212, -65.033465, -65.1135912, -65.1018904, -65.0391091, -64.9484026, -64.8455287, -64.740033 };

    auto dec_gkbar_0036 = arb::decor{}
        .paint(arb::reg::tagged(1), arb::density("hh", {{"gkbar", 0.036}}))
        .paint(arb::reg::tagged(2), arb::density("pas"))
        .place(arb::ls::location(0, 0.5), arb::i_clamp::box(10_ms, 20_ms, 100_pA), "ic1")
        ;
    auto dec_gkbar_0008 = arb::decor{}
        .paint(arb::reg::tagged(1), arb::density("hh", {{"gkbar", 0.008}}))
        .paint(arb::reg::tagged(2), arb::density("pas"))
        .place(arb::ls::location(0, 0.5), arb::i_clamp::box(10_ms, 20_ms, 100_pA), "ic1")
        ;

    auto ctx = arb::make_context({arbenv::default_concurrency(), with_gpu});

    // results must be invariant under the group size, even if it doesn't divide into N
    for (size_t g_size = N; g_size <= N; ++g_size) {
        // ... and the gid targeted
        for (size_t gid = 1; gid < N; ++gid) {
            auto rec_0008 = cable_recipe{dec_gkbar_0008};
            auto sim_0008 = arb::simulation{rec_0008,
                                            ctx,
                                            partition_load_balance(rec_0008,
                                                                   ctx,
                                                                   {{arb::cell_kind::cable, arb::partition_hint{.cpu_group_size=g_size}}})};
            sim_0008.add_sampler(arb::all_probes, arb::regular_schedule(dt*arb::units::ms), sampler);
            // now re-write the gkbar parameter
            sim_0008.edit_cell(gid,
                               arb::cable_cell_editor {
                                   .on_density = arb::density_editor([] (const auto&, const auto& what, const auto&) -> arb::parameter_map {
                                       if (what == "hh") return {{"gkbar", 0.036}};
                                       return {};
                                   })});
            sim_0008.run(T*arb::units::ms, dt*arb::units::ms);
            auto samples_0008 = sample_values;
            for (auto& row: sample_values) std::fill(row.begin(), row.end(), 0.0);
            
            
            auto rec_0036 = cable_recipe{dec_gkbar_0036};
            auto sim_0036 = arb::simulation{rec_0036,
                                            ctx,
                                            partition_load_balance(rec_0036,
                                                                   ctx,
                                                                   {{arb::cell_kind::cable, arb::partition_hint{.cpu_group_size=g_size}}})};
            sim_0036.add_sampler(arb::all_probes, arb::regular_schedule(dt*arb::units::ms), sampler);
            // now re-write the gkbar parameter
            sim_0036.edit_cell(gid,
                               arb::cable_cell_editor {
                                   .on_density = arb::density_editor([] (const auto&, const auto& what, const auto&) -> arb::parameter_map {
                                       if (what == "hh") return {{"gkbar", 0.008}};
                                       return {};
                                   })});            
            sim_0036.run(T*arb::units::ms, dt*arb::units::ms);
            auto samples_0036 = sample_values;
            for (auto& row: sample_values) std::fill(row.begin(), row.end(), 0.0);
            
            // check each unaltered cell against each other 
            for (unsigned col = 0; col < N; ++col) {
                if (col == gid) continue;
                EXPECT_TRUE(all_near(unedited, samples_0036, col, eps));
                EXPECT_TRUE(all_near(  edited, samples_0008, col, eps));
            }
            // now check the edited ones
            EXPECT_TRUE(all_near(unedited, samples_0008, gid, eps));
            EXPECT_TRUE(all_near(  edited, samples_0036, gid, eps));
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

    std::vector unedited = { -65.000000, -65.976650, -66.650927, -67.003375, -67.167843, -67.211650, -67.190473, -67.136324, -67.069902, -67.002777, -66.941313, -65.466084, -64.416894,  -63.769651, -63.388147, -63.232318, -63.250644, -63.392280, -63.602123, -63.830035, -64.038928, -64.208130, -64.331221, -64.411239, -64.455972, -64.474584, -64.475653, -64.466276, -64.451815, -64.436002, -64.421192, -65.752674, -66.673119, -67.139232, -67.345711, -67.391215, -67.353235, -67.274783, -67.182795, -67.091783};
    std::vector   edited = { -65.000000, -68.3113158, -69.2069298, -69.3547088, -69.3686045, -69.338811, -69.3119782, -69.2868814, -69.2672765, -69.2510752, -69.2381611, -68.6364915, -68.4719094, -68.4352472, -68.4243892, -68.4229709, -68.4226988, -68.4231265, -68.4234685, -68.423814, -68.4240821, -68.4243069, -68.4244849, -68.4246288, -68.4247431, -68.4248342, -68.4249062, -68.4249629, -68.4250073, -68.4250417, -68.4250682, -69.0158919, -69.172337, -69.2018512, -69.2068637, -69.2035358, -69.2000019, -69.1964224, -69.1935063, -69.1910237 };

    auto dec = arb::decor{}
        .paint(arb::reg::tagged(1), arb::density("hh", {{"gkbar", 0.036}}))
        .paint(arb::reg::tagged(2), arb::density("pas"))
        .place(arb::ls::location(0, 0.5), arb::i_clamp::box(10_ms, 20_ms, 100_pA), "ic1")
        ;
    auto ctx = arb::make_context({arbenv::default_concurrency(), with_gpu});
    auto rec = cable_recipe{dec};

    // results must be invariant under the group size, even if it doesn't divide into N
    for (size_t g_size = 1; g_size <= N; ++g_size) {
        // ... and the gid targeted
        for (size_t gid = 0; gid < N; ++gid) {
            auto sim = arb::simulation{rec,
                                       ctx,
                                       partition_load_balance(rec,
                                                              ctx,
                                                              {{arb::cell_kind::cable, arb::partition_hint{.cpu_group_size=g_size}}})};
            sim.add_sampler(arb::all_probes, arb::regular_schedule(dt*arb::units::ms), sampler);
            sim.edit_cell(gid,
                          arb::cable_cell_editor {
                              .on_density = [] (const auto&, const auto& what, const auto&) -> arb::parameter_map {
                                  if (what == "pas") return {{"g", 0.008}};
                                  return {};
                              }});

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

struct cable_gen_recipe: cable_recipe {
    cable_gen_recipe(const arb::decor& dec): cable_recipe(dec) {}
    
    std::vector<arb::event_generator> event_generators(arb::cell_gid_type) const override {
        return {arb::poisson_generator({"syn"}, 0.005, 0_ms, 1.0_kHz, 42)};
    }
};

TEST(edit_cable, expsyn) {
    result_t sample_values;
    sample_values.resize(n_step);
    auto sampler = [&sample_values](arb::probe_metadata pm, std::size_t n, const arb::sample_record* samples) {
        auto gid = pm.id.gid;
        for (std::size_t ix = 0; ix < n; ++ix) {
            sample_values[ix][gid] = *arb::util::any_cast<const double*>(samples[ix].data);
        }
    };

    std::vector exp_20 = { -65.0, -54.7514995, -42.2686825, 0.137012342, 21.8149167, -45.364232, -67.9129598, -72.1109645, -68.779846, -64.7865025, -64.9626734, -61.1617474, -59.1857019, -60.6333545, -57.8448538, -55.8318071, -56.8708576, -55.3365449, -49.8457899, -41.2636889, -20.9274785, -2.98642657, -39.2739903, -64.2217407, -71.2949142, -71.7633758, -69.3955239, -66.2353185, -60.7635546, -55.6521287, -50.6940555, -41.8170467, -21.6652507, -1.49099787, -38.0130863, -63.7076861, -70.3662875, -70.8823108, -71.0628905, -65.465745, };
    std::vector exp_40 = { -65.0, -54.7514995, -41.1813592, 3.78404154, 20.0808805, -46.28961, -67.2793905, -69.8274219, -65.9098412, -61.2754815, -60.0784287, -56.4210018, -53.7311027, -53.206293, -51.0969989, -48.7490513, -47.3695081, -45.168154, -41.1002217, -36.1433987, -33.5465413, -39.3917815, -52.7034955, -64.0943558, -64.1922785, -63.7424232, -61.3787013, -58.6864932, -54.1679162, -48.8244327, -40.8628796, -24.7089839, -11.9162012, -37.5491484, -59.5190358, -68.2175105, -65.258128, -64.1614843, -63.660979, -58.9115015, };
    
    auto dec_tau_20 = arb::decor{}
        .paint(arb::reg::tagged(1), arb::density("hh", {{"gkbar", 0.036}}))
        .paint(arb::reg::tagged(2), arb::density("pas"))
        .place(arb::ls::location(0, 0.5), arb::synapse("expsyn", {{"tau", 2.0}}), "syn")
        ;
    auto dec_tau_40 = arb::decor{}
        .paint(arb::reg::tagged(1), arb::density("hh", {{"gkbar", 0.036}}))
        .paint(arb::reg::tagged(2), arb::density("pas"))
        .place(arb::ls::location(0, 0.5), arb::synapse("expsyn", {{"tau", 4.0}}), "syn")
        ;

    auto ctx = arb::make_context({arbenv::default_concurrency(), with_gpu});

    // results must be invariant under the group size, even if it doesn't divide into N
    for (size_t g_size = N; g_size > 1; --g_size) {
        // ... and the gid targeted
        for (size_t gid = 0; gid < N; ++gid) {
            auto rec_20 = cable_gen_recipe{dec_tau_20};
            auto sim_20 = arb::simulation{rec_20,
                                            ctx,
                                            partition_load_balance(rec_20,
                                                                   ctx,
                                                                   {{arb::cell_kind::cable, arb::partition_hint{.cpu_group_size=g_size}}})};
            sim_20.edit_cell(gid,
                               arb::cable_cell_editor {
                                   .on_synapse = [] (const auto&, const auto& what, const auto&) -> arb::parameter_map {
                                       if (what == "expsyn") return {{"tau", 4.0}};
                                       return {};
                                   }});
            sim_20.add_sampler(arb::all_probes, arb::regular_schedule(dt*arb::units::ms), sampler);
            sim_20.run(T*arb::units::ms, dt*arb::units::ms);
            auto samples_20 = sample_values;
            for (auto& row: sample_values) std::fill(row.begin(), row.end(), 0.0);

            auto rec_40 = cable_gen_recipe{dec_tau_40};
            auto sim_40 = arb::simulation{rec_40,
                                          ctx,
                                          partition_load_balance(rec_40,
                                                                 ctx,
                                                                 {{arb::cell_kind::cable, arb::partition_hint{.cpu_group_size=g_size}}})};
            sim_40.edit_cell(gid,
                               arb::cable_cell_editor {
                                   .on_synapse = [] (const auto&, const auto& what, const auto&) -> arb::parameter_map {
                                       if (what == "expsyn") return {{"tau", 2.0}};
                                       return {};
                                   }});
            sim_40.add_sampler(arb::all_probes, arb::regular_schedule(dt*arb::units::ms), sampler);
            sim_40.run(T*arb::units::ms, dt*arb::units::ms);
            auto samples_40 = sample_values;
            for (auto& row: sample_values) std::fill(row.begin(), row.end(), 0.0);
            
            // check each unaltered cell against each other 
            for (unsigned col = 0; col < N; ++col) {
                if (col == gid) continue;
                EXPECT_TRUE(all_near(exp_20, samples_20, col, eps));
                EXPECT_TRUE(all_near(exp_40, samples_40, col, eps));
            }
            // now check the edited ones
            EXPECT_TRUE(all_near(exp_20, samples_40, gid, eps));
            EXPECT_TRUE(all_near(exp_40, samples_20, gid, eps));            
        }
    }
}

// NOTE just tests _that_ we can, not correctness. Should probably change that.
TEST(edit_cable, can_edit_derived) {
    auto dec = arb::decor{}
        .paint(arb::reg::tagged(1), arb::density("hh", {{"gkbar", 0.036}}))
        .paint(arb::reg::tagged(2), arb::density("pas/e=-80", {{"g", 0.1}}))
        .paint(arb::reg::tagged(2), arb::density("pas/e=-70", {{"g", 0.2}}))
        .paint(arb::reg::tagged(2), arb::density("pas/e=-60", {{"g", 0.3}}))
        .place(arb::ls::location(0, 0.5), arb::i_clamp::box(10_ms, 20_ms, 100_pA), "ic1")
        ;
    auto rec = cable_recipe{dec};
    auto sim = arb::simulation{rec};
    sim.edit_cell(1,
                  arb::cable_cell_editor {
                      .on_density = [] (const auto&, const auto& what, const auto&) -> arb::parameter_map {
                          if (what == "pas/e=-70") return {{"g", 0.008}};
                          return {};
                      }});
    sim.run(T*arb::units::ms, dt*arb::units::ms);
}
