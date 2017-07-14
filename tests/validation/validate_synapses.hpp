#include <json/json.hpp>

#include <cell.hpp>
#include <cell_group.hpp>
#include <fvm_multicell.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <simple_sampler.hpp>
#include <util/path.hpp>

#include "../gtest.h"

#include "../test_common_cells.hpp"
#include "convergence_test.hpp"
#include "trace_analysis.hpp"
#include "validation_data.hpp"

void run_synapse_test(
    const char* syn_type,
    const nest::mc::util::path& ref_data_path,
    nest::mc::backend_policy backend,
    float t_end=70.f,
    float dt=0.001)
{
    using namespace nest::mc;

    auto max_ncomp = g_trace_io.max_ncomp();
    nlohmann::json meta = {
        {"name", "membrane voltage"},
        {"model", syn_type},
        {"sim", "nestmc"},
        {"units", "mV"},
        {"backend_policy", to_string(backend)}
    };

    cell c = make_cell_ball_and_stick(false); // no stimuli
    parameter_list syn_default(syn_type);
    c.add_synapse({1, 0.5}, syn_default);
    add_common_voltage_probes(c);

    // injected spike events
    std::vector<postsynaptic_spike_event> synthetic_events = {
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

    convergence_test_runner<int> runner("ncomp", samplers, meta);
    runner.load_reference_data(ref_data_path);

    node_description nd(1, backend==backend_policy::gpu? 1: 0);
    for (int ncomp = 10; ncomp<max_ncomp; ncomp*=2) {
        c.cable(1)->set_compartments(ncomp);
        domain_decomposition decomp(singleton_recipe{c}, nd);
        model m(singleton_recipe{c}, decomp);
        m.group(0).enqueue_events(synthetic_events);

        runner.run(m, ncomp, t_end, dt, exclude);
    }
    runner.report();
    runner.assert_all_convergence();
}
