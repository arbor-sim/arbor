#include "../gtest.h"

#include <string>

#include <nlohmann/json.hpp>

#include <arbor/context.hpp>
#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

#include "convergence_test.hpp"
#include "trace_analysis.hpp"
#include "util.hpp"
#include "validation_data.hpp"

void run_kinetic_dt(
    const arb::context& context,
    arb::cable_cell& c,
    arb::cell_probe_address probe,
    float t_end,
    nlohmann::json meta,
    const std::string& ref_file)
{
    using namespace arb;

    float sample_dt = g_trace_io.sample_dt();

    cable1d_recipe rec{c};
    rec.add_probe(0, 0, probe);

    probe_label plabels[1] = {{"soma.mid", {0u, 0u}}};

    meta["sim"] = "arbor";
    meta["backend_kind"] = arb::has_gpu(context)? "gpu": "multicore";

    convergence_test_runner<float> runner("dt", plabels, meta);
    runner.load_reference_data(ref_file);

    auto decomp = partition_load_balance(rec, context);
    simulation sim(rec, decomp, context);

    auto exclude = stimulus_ends(c);

    // use dt = 0.05, 0.02, 0.01, 0.005, 0.002,  ...
    double max_oo_dt = std::round(1.0/g_trace_io.min_dt());
    for (double base = 100; ; base *= 10) {
        for (double multiple: {5., 2., 1.}) {
            double oo_dt = base/multiple;
            if (oo_dt>max_oo_dt) goto end;

            sim.reset();
            float dt = float(1./oo_dt);
            runner.run(sim, dt, sample_dt, t_end, dt, exclude);
        }
    }

end:
    runner.report();
    runner.assert_all_convergence();
}

void validate_kinetic_kin1(const arb::context& ctx) {
    using namespace arb;

    // 20 µm diameter soma with single mechanism, current probe
    cable_cell c;
    auto soma = c.add_soma(10);
    soma->add_mechanism("test_kin1");
    cell_probe_address probe{{0, 0.5}, cell_probe_address::membrane_current};

    nlohmann::json meta = {
        {"model", "test_kin1"},
        {"name", "membrane current"},
        {"units", "nA"}
    };

    run_kinetic_dt(ctx, c, probe, 100.f, meta, "numeric_kin1.json");
}

void validate_kinetic_kinlva(const arb::context& ctx) {
    using namespace arb;

    // 20 µm diameter soma with single mechanism, current probe
    cable_cell c;
    auto soma = c.add_soma(10);
    c.add_stimulus({0,0.5}, {20., 130., -0.025});
    soma->add_mechanism("test_kinlva");
    cell_probe_address probe{{0, 0.5}, cell_probe_address::membrane_voltage};

    nlohmann::json meta = {
        {"model", "test_kinlva"},
        {"name", "membrane voltage"},
        {"units", "mV"}
    };

    run_kinetic_dt(ctx, c, probe, 300.f, meta, "numeric_kinlva.json");
}


using namespace arb;

TEST(kinetic, kin1_numeric_ref) {
    proc_allocation resources;
    {
        auto ctx = make_context(resources);
        validate_kinetic_kin1(ctx);
    }
    if (resources.has_gpu()) {
        resources.gpu_id = -1;
        auto ctx = make_context(resources);
        validate_kinetic_kin1(ctx);
    }
}

TEST(kinetic, kinlva_numeric_ref) {
    proc_allocation resources;
    {
        auto ctx = make_context(resources);
        validate_kinetic_kinlva(ctx);
    }
    if (resources.has_gpu()) {
        resources.gpu_id = -1;
        auto ctx = make_context(resources);
        validate_kinetic_kinlva(ctx);
    }
}
