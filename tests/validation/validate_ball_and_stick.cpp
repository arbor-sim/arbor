#include <json/json.hpp>

#include <cell.hpp>
#include <common_types.hpp>
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

template <
    typename lowered_cell,
    typename SamplerInfoSeq
>
void run_ncomp_convergence_test(
    const char* model_name,
    const util::path& ref_data_path,
    const cell& c,
    SamplerInfoSeq& samplers,
    float t_end=100.f)
{
    auto max_ncomp = g_trace_io.max_ncomp();
    auto dt = g_trace_io.min_dt();

    nlohmann::json meta = {
        {"name", "membrane voltage"},
        {"model", model_name},
        {"dt", dt},
        {"sim", "nestmc"},
        {"units", "mV"}
    };

    auto exclude = stimulus_ends(c);

    convergence_test_runner<int> R("ncomp", samplers, meta);
    R.load_reference_data(ref_data_path);

    for (int ncomp = 10; ncomp<max_ncomp; ncomp*=2) {
        for (auto& seg: c.segments()) {
            if (!seg->is_soma()) {
                seg->set_compartments(ncomp);
            }
        }
        model<lowered_cell> m(singleton_recipe{c});

        R.run(m, ncomp, t_end, dt, exclude);
    }
    R.report();
    R.assert_all_convergence();
}

TEST(ball_and_taper, neuron_ref) {
    using lowered_cell = fvm::fvm_multicell<double, cell_local_size_type>;

    cell c = make_cell_ball_and_stick();
    add_common_voltage_probes(c);

    float sample_dt = 0.025f;
    sampler_info samplers[] = {
        {"soma.mid", {0u, 0u}, simple_sampler(sample_dt)},
        {"dend.mid", {0u, 1u}, simple_sampler(sample_dt)},
        {"dend.end", {0u, 2u}, simple_sampler(sample_dt)}
    };

    run_ncomp_convergence_test<lowered_cell>(
        "ball_and_stick",
        "neuron_ball_and_stick.json",
        c,
        samplers);
}

TEST(ball_and_3stick, neuron_ref) {
    using lowered_cell = fvm::fvm_multicell<double, cell_local_size_type>;

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

    run_ncomp_convergence_test<lowered_cell>(
        "ball_and_3stick",
        "neuron_ball_and_3stick.json",
        c,
        samplers);
}

TEST(rallpack1, numeric_ref) {
    using lowered_cell = fvm::fvm_multicell<double, cell_local_size_type>;

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

    run_ncomp_convergence_test<lowered_cell>(
        "rallpack1",
        "numeric_rallpack1.json",
        c,
        samplers,
        250.f);
}

template <typename Policy>
using lowered_cell_div = fvm::fvm_multicell<double, cell_local_size_type, Policy>;

TEST(ball_and_squiggle, neuron_ref) {
    cell c = make_cell_ball_and_squiggle();
    add_common_voltage_probes(c);

    float sample_dt = 0.025f;
    sampler_info samplers[] = {
        {"soma.mid", {0u, 0u}, simple_sampler(sample_dt)},
        {"dend.mid", {0u, 1u}, simple_sampler(sample_dt)},
        {"dend.end", {0u, 2u}, simple_sampler(sample_dt)}
    };

    run_ncomp_convergence_test<lowered_cell_div<div_compartment_sampler>>(
        "ball_and_squiggle_sampler",
        "neuron_ball_and_squiggle.json",
        c,
        samplers);

    run_ncomp_convergence_test<lowered_cell_div<div_compartment_integrator>>(
        "ball_and_squiggle_integrator",
        "neuron_ball_and_squiggle.json",
        c,
        samplers);
}
