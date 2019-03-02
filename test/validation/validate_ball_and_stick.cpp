#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <sup/path.hpp>

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

#include "convergence_test.hpp"
#include "trace_analysis.hpp"
#include "util.hpp"
#include "validation_data.hpp"

#include "../gtest.h"

using namespace arb;

struct probe_point {
    const char* label;
    segment_location where;
};

template <typename ProbePointSeq>
void run_ncomp_convergence_test(
    const char* model_name,
    const sup::path& ref_data_path,
    context& context,
    const cable_cell& c,
    ProbePointSeq& probe_points,
    float t_end=100.f)
{
    using namespace arb;

    auto max_ncomp = g_trace_io.max_ncomp();
    auto dt = g_trace_io.min_dt();
    auto sample_dt = g_trace_io.sample_dt();

    nlohmann::json meta = {
        {"name", "membrane voltage"},
        {"model", model_name},
        {"dt", dt},
        {"sim", "arbor"},
        {"units", "mV"},
        {"backend_kind", (has_gpu(context)? "gpu": "multicore")}
    };

    auto exclude = stimulus_ends(c);

    auto n_probe = size(probe_points);
    std::vector<probe_label> plabels;
    plabels.reserve(n_probe);
    for (unsigned i = 0; i<n_probe; ++i) {
        plabels.push_back(probe_label{probe_points[i].label, {0u, i}});
    }

    convergence_test_runner<int> runner("ncomp", plabels, meta);
    runner.load_reference_data(ref_data_path);

    for (int ncomp = 10; ncomp<max_ncomp; ncomp*=2) {
        for (auto& seg: c.segments()) {
            if (!seg->is_soma()) {
                seg->set_compartments(ncomp);
            }
        }
        cable1d_recipe rec{c};
        for (const auto& p: probe_points) {
            rec.add_probe(0, 0, cell_probe_address{p.where, cell_probe_address::membrane_voltage});
        }

        auto decomp = partition_load_balance(rec, context);
        simulation sim(rec, decomp, context);

        runner.run(sim, ncomp, sample_dt, t_end, dt, exclude);
    }
    runner.report();
    runner.assert_all_convergence();
}

void validate_ball_and_stick(context& ctx) {
    using namespace arb;

    cable_cell c = make_cell_ball_and_stick();
    probe_point points[] = {
        {"soma.mid", {0u, 0.5}},
        {"dend.mid", {1u, 0.5}},
        {"dend.end", {1u, 1.0}}
    };

    run_ncomp_convergence_test(
        "ball_and_stick",
        "neuron_ball_and_stick.json",
        ctx,
        c,
        points);
}

void validate_ball_and_taper(context& ctx) {
    using namespace arb;

    cable_cell c = make_cell_ball_and_taper();
    probe_point points[] = {
        {"soma.mid",  {0u, 0.5}},
        {"taper.mid", {1u, 0.5}},
        {"taper.end", {1u, 1.0}}
    };

    run_ncomp_convergence_test(
        "ball_and_taper",
        "neuron_ball_and_taper.json",
        ctx,
        c,
        points);
}

void validate_ball_and_3stick(context& ctx) {
    using namespace arb;

    cable_cell c = make_cell_ball_and_3stick();
    probe_point points[] = {
        {"soma.mid",  {0u, 0.5}},
        {"dend1.mid", {1u, 0.5}},
        {"dend1.end", {1u, 1.0}},
        {"dend2.mid", {2u, 0.5}},
        {"dend2.end", {2u, 1.0}},
        {"dend3.mid", {3u, 0.5}},
        {"dend3.end", {3u, 1.0}}
    };

    run_ncomp_convergence_test(
        "ball_and_3stick",
        "neuron_ball_and_3stick.json",
        ctx,
        c,
        points);
}

void validate_rallpack1(context& ctx) {
    using namespace arb;

    cable_cell c = make_cell_simple_cable();
    probe_point points[] = {
        {"cable.x0.0", {1u, 0.0}},
        {"cable.x0.3", {1u, 0.3}},
        {"cable.x1.0", {1u, 1.0}}
    };

    run_ncomp_convergence_test(
        "rallpack1",
        "numeric_rallpack1.json",
        ctx,
        c,
        points,
        250.f);
}

void validate_ball_and_squiggle(context& ctx) {
    using namespace arb;

    cable_cell c = make_cell_ball_and_squiggle();
    probe_point points[] = {
        {"soma.mid", {0u, 0.5}},
        {"dend.mid", {1u, 0.5}},
        {"dend.end", {1u, 1.0}}
    };

#if 0
    // *temporarily* disabled: compartment division policy will
    // be moved into backend policy classes.

    run_ncomp_convergence_test<lowered_cell_div<div_compartment_sampler>>(
        "ball_and_squiggle_sampler",
        "neuron_ball_and_squiggle.json",
        c,
        samplers);
#endif

    run_ncomp_convergence_test(
        "ball_and_squiggle_integrator",
        "neuron_ball_and_squiggle.json",
        ctx,
        c,
        points);
}

TEST(ball_and_stick, neuron_ref) {
    proc_allocation resources;
    {
        auto ctx = make_context(resources);
        validate_ball_and_stick(ctx);
    }
    if (resources.has_gpu()) {
        resources.gpu_id = -1;
        auto ctx = make_context(resources);
        validate_ball_and_stick(ctx);
    }
}

TEST(ball_and_taper, neuron_ref) {
    proc_allocation resources;
    {
        auto ctx = make_context(resources);
        validate_ball_and_taper(ctx);
    }
    if (resources.has_gpu()) {
        resources.gpu_id = -1;
        auto ctx = make_context(resources);
        validate_ball_and_taper(ctx);
    }
}

TEST(ball_and_3stick, neuron_ref) {
    proc_allocation resources;
    {
        auto ctx = make_context(resources);
        validate_ball_and_3stick(ctx);
    }
    if (resources.has_gpu()) {
        resources.gpu_id = -1;
        auto ctx = make_context(resources);
        validate_ball_and_3stick(ctx);
    }
}

TEST(rallpack1, numeric_ref) {
    proc_allocation resources;
    {
        auto ctx = make_context(resources);
        validate_rallpack1(ctx);
    }
    if (resources.has_gpu()) {
        resources.gpu_id = -1;
        auto ctx = make_context(resources);
        validate_rallpack1(ctx);
    }
}

TEST(ball_and_squiggle, neuron_ref) {
    proc_allocation resources;
    {
        auto ctx = make_context(resources);
        validate_ball_and_squiggle(ctx);
    }
    if (resources.has_gpu()) {
        resources.gpu_id = -1;
        auto ctx = make_context(resources);
        validate_ball_and_squiggle(ctx);
    }
}
