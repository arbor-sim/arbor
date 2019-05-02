#include <nlohmann/json.hpp>

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <sup/path.hpp>


#include "../gtest.h"

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

#include "convergence_test.hpp"
#include "trace_analysis.hpp"
#include "util.hpp"
#include "validation_data.hpp"

using namespace arb;

void run_synapse_test(
    const char* syn_type,
    const sup::path& ref_data_path,
    const context& context,
    float t_end=70.f,
    float dt=0.001)
{
    auto max_ncomp = g_trace_io.max_ncomp();
    nlohmann::json meta = {
        {"name", "membrane voltage"},
        {"model", syn_type},
        {"sim", "arbor"},
        {"units", "mV"},
        {"backend_kind", arb::has_gpu(context)? "gpu": "multicore"}
    };

    cable_cell c = make_cell_ball_and_stick(false); // no stimuli
    mechanism_desc syn_default(syn_type);
    c.add_synapse({1, 0.5}, syn_default);

    // injected spike events
    std::vector<spike_event> synthetic_events = {
        {{0u, 0u}, 10.0, 0.04},
        {{0u, 0u}, 20.0, 0.04},
        {{0u, 0u}, 40.0, 0.04}
    };

    // exclude points of discontinuity from linf analysis
    std::vector<float> exclude = {10.f, 20.f, 40.f};

    float sample_dt = g_trace_io.sample_dt();
    probe_label plabels[3] = {
        {"soma.mid", {0u, 0u}},
        {"dend.mid", {0u, 1u}},
        {"dend.end", {0u, 2u}}
    };

    convergence_test_runner<int> runner("ncomp", plabels, meta);
    runner.load_reference_data(ref_data_path);

    for (int ncomp = 10; ncomp<max_ncomp; ncomp*=2) {
        c.cable(1)->set_compartments(ncomp);

        cable1d_recipe rec{c};
        // soma.mid
        rec.add_probe(0, 0, cell_probe_address{{0, 0.5}, cell_probe_address::membrane_voltage});
        // dend.mid
        rec.add_probe(0, 0, cell_probe_address{{1, 0.5}, cell_probe_address::membrane_voltage});
        // dend.end
        rec.add_probe(0, 0, cell_probe_address{{1, 1.0}, cell_probe_address::membrane_voltage});

        auto decomp = partition_load_balance(rec, context);
        simulation sim(rec, decomp, context);

        sim.inject_events(synthetic_events);

        runner.run(sim, ncomp, sample_dt, t_end, dt, exclude);
    }
    runner.report();
    runner.assert_all_convergence();
}

TEST(simple_synapse, expsyn_neuron_ref) {
    proc_allocation resources;
    {
        auto ctx = make_context(resources);
        SCOPED_TRACE("expsyn-multicore");
        run_synapse_test("expsyn", "neuron_simple_exp_synapse.json", ctx);
    }
    if (resources.has_gpu()) {
        resources.gpu_id = -1;
        auto ctx = make_context(resources);
        SCOPED_TRACE("expsyn-gpu");
        run_synapse_test("expsyn", "neuron_simple_exp_synapse.json", ctx);
    }
}

TEST(simple_synapse, exp2syn_neuron_ref) {
    proc_allocation resources;
    {
        auto ctx = make_context(resources);
        SCOPED_TRACE("exp2syn-multicore");
        run_synapse_test("exp2syn", "neuron_simple_exp_synapse.json", ctx);
    }
    if (resources.has_gpu()) {
        resources.gpu_id = -1;
        auto ctx = make_context(resources);
        SCOPED_TRACE("exp2syn-gpu");
        run_synapse_test("exp2syn", "neuron_simple_exp_synapse.json", ctx);
    }
}
