#include <json/json.hpp>

#include <cell.hpp>
#include <fvm_multicell.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <simple_sampler.hpp>
#include <util/path.hpp>

#include "gtest.h"

#include "../test_common_cells.hpp"
#include "convergence_test.hpp"
#include "trace_analysis.hpp"
#include "validation_data.hpp"

using namespace nest::mc;

void run_synapse_test(
    const char* syn_type,
    const util::path& ref_data_path,
    float t_end=70.f,
    float dt=0.001)
{
    using lowered_cell = fvm::fvm_multicell<double, cell_local_size_type>;

    auto max_ncomp = g_trace_io.max_ncomp();
    nlohmann::json meta = {
        {"name", "membrane voltage"},
        {"model", syn_type},
        {"sim", "nestmc"},
        {"units", "mV"}
    };

    cell c = make_cell_ball_and_stick(false); // no stimuli
    parameter_list syn_default(syn_type);
    c.add_synapse({1, 0.5}, syn_default);
    add_common_voltage_probes(c);

    // injected spike events
    postsynaptic_spike_event<float> synthetic_events[] = {
        {{0u, 0u}, 10.0, 0.04},
        {{0u, 0u}, 20.0, 0.04},
        {{0u, 0u}, 40.0, 0.04}
    };

    // exclude points of discontinuity from linf analysis
    std::vector<float> exclude = {10.f, 20.f, 40.f};

    float sample_dt = 0.025f;
    sampler_info samplers[] = {
        {"soma.mid", {0u, 0u}, simple_sampler(sample_dt)},
        {"dend.mid", {0u, 1u}, simple_sampler(sample_dt)},
        {"dend.end", {0u, 2u}, simple_sampler(sample_dt)}
    };

    convergence_test_runner<int> R("ncomp", samplers, meta);
    R.load_reference_data(ref_data_path);

    for (int ncomp = 10; ncomp<max_ncomp; ncomp*=2) {
        c.cable(1)->set_compartments(ncomp);
        model<lowered_cell> m(singleton_recipe{c});
        m.group(0).enqueue_events(synthetic_events);

        R.run(m, ncomp, t_end, dt, exclude);
    }
    R.report();
    R.assert_all_convergence();
}

TEST(simple_synapse, expsyn_neuron_ref) {
    SCOPED_TRACE("expsyn");
    run_synapse_test("expsyn", "neuron_simple_exp_synapse.json");
}

TEST(simple_synapse, exp2syn_neuron_ref) {
    SCOPED_TRACE("exp2syn");
    run_synapse_test("exp2syn", "neuron_simple_exp2_synapse.json");
}

