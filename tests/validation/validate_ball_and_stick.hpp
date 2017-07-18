#include <json/json.hpp>

#include <cell.hpp>
#include <common_types.hpp>
#include <fvm_multicell.hpp>
#include <hardware/node.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <simple_sampler.hpp>
#include <util/path.hpp>

#include "../gtest.h"

#include "../test_common_cells.hpp"
#include "convergence_test.hpp"
#include "trace_analysis.hpp"
#include "validation_data.hpp"

template <typename SamplerInfoSeq>
void run_ncomp_convergence_test(
    const char* model_name,
    const nest::mc::util::path& ref_data_path,
    nest::mc::backend_kind backend,
    const nest::mc::cell& c,
    SamplerInfoSeq& samplers,
    float t_end=100.f)
{
    using namespace nest::mc;

    auto max_ncomp = g_trace_io.max_ncomp();
    auto dt = g_trace_io.min_dt();

    nlohmann::json meta = {
        {"name", "membrane voltage"},
        {"model", model_name},
        {"dt", dt},
        {"sim", "nestmc"},
        {"units", "mV"},
        {"backend_kind", to_string(backend)}
    };

    auto exclude = stimulus_ends(c);

    convergence_test_runner<int> runner("ncomp", samplers, meta);
    runner.load_reference_data(ref_data_path);

    for (int ncomp = 10; ncomp<max_ncomp; ncomp*=2) {
        for (auto& seg: c.segments()) {
            if (!seg->is_soma()) {
                seg->set_compartments(ncomp);
            }
        }
        hw::node nd(1, backend==backend_kind::gpu? 1: 0);
        domain_decomposition decomp(singleton_recipe{c}, nd);
        model m(singleton_recipe{c}, decomp);

        runner.run(m, ncomp, t_end, dt, exclude);
    }
    runner.report();
    runner.assert_all_convergence();
}

void validate_ball_and_stick(nest::mc::backend_kind backend) {
    using namespace nest::mc;

    cell c = make_cell_ball_and_stick();
    add_common_voltage_probes(c);

    float sample_dt = 0.025f;
    sampler_info samplers[] = {
        {"soma.mid", {0u, 0u}, simple_sampler(sample_dt)},
        {"dend.mid", {0u, 1u}, simple_sampler(sample_dt)},
        {"dend.end", {0u, 2u}, simple_sampler(sample_dt)}
    };

    run_ncomp_convergence_test(
        "ball_and_stick",
        "neuron_ball_and_stick.json",
        backend,
        c,
        samplers);
}

void validate_ball_and_taper(nest::mc::backend_kind backend) {
    using namespace nest::mc;

    cell c = make_cell_ball_and_taper();
    add_common_voltage_probes(c);

    float sample_dt = 0.025f;
    sampler_info samplers[] = {
        {"soma.mid", {0u, 0u}, simple_sampler(sample_dt)},
        {"taper.mid", {0u, 1u}, simple_sampler(sample_dt)},
        {"taper.end", {0u, 2u}, simple_sampler(sample_dt)}
    };

    run_ncomp_convergence_test(
        "ball_and_taper",
        "neuron_ball_and_taper.json",
        backend,
        c,
        samplers);
}

void validate_ball_and_3stick(nest::mc::backend_kind backend) {
    using namespace nest::mc;

    cell c = make_cell_ball_and_3stick();
    add_common_voltage_probes(c);

    float sample_dt = 0.025f;
    sampler_info samplers[] = {
        {"soma.mid",  {0u, 0u}, simple_sampler(sample_dt)},
        {"dend1.mid", {0u, 1u}, simple_sampler(sample_dt)},
        {"dend1.end", {0u, 2u}, simple_sampler(sample_dt)},
        {"dend2.mid", {0u, 3u}, simple_sampler(sample_dt)},
        {"dend2.end", {0u, 4u}, simple_sampler(sample_dt)},
        {"dend3.mid", {0u, 5u}, simple_sampler(sample_dt)},
        {"dend3.end", {0u, 6u}, simple_sampler(sample_dt)}
    };

    run_ncomp_convergence_test(
        "ball_and_3stick",
        "neuron_ball_and_3stick.json",
        backend,
        c,
        samplers);
}

void validate_rallpack1(nest::mc::backend_kind backend) {
    using namespace nest::mc;

    cell c = make_cell_simple_cable();

    // three probes: left end, 30% along, right end.
    c.add_probe({{1, 0.0}, probeKind::membrane_voltage});
    c.add_probe({{1, 0.3}, probeKind::membrane_voltage});
    c.add_probe({{1, 1.0}, probeKind::membrane_voltage});

    float sample_dt = 0.025f;
    sampler_info samplers[] = {
        {"cable.x0.0", {0u, 0u}, simple_sampler(sample_dt)},
        {"cable.x0.3", {0u, 1u}, simple_sampler(sample_dt)},
        {"cable.x1.0", {0u, 2u}, simple_sampler(sample_dt)},
    };

    run_ncomp_convergence_test(
        "rallpack1",
        "numeric_rallpack1.json",
        backend,
        c,
        samplers,
        250.f);
}

void validate_ball_and_squiggle(nest::mc::backend_kind backend) {
    using namespace nest::mc;

    cell c = make_cell_ball_and_squiggle();
    add_common_voltage_probes(c);

    float sample_dt = 0.025f;
    sampler_info samplers[] = {
        {"soma.mid", {0u, 0u}, simple_sampler(sample_dt)},
        {"dend.mid", {0u, 1u}, simple_sampler(sample_dt)},
        {"dend.end", {0u, 2u}, simple_sampler(sample_dt)}
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
        backend,
        c,
        samplers);
}
