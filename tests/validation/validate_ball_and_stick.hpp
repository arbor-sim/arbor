#include <json/json.hpp>

#include <cell.hpp>
#include <common_types.hpp>
#include <fvm_multicell.hpp>
#include <load_balance.hpp>
#include <hardware/node_info.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <segment.hpp>
#include <simple_sampler.hpp>
#include <util/meta.hpp>
#include <util/path.hpp>

#include "../gtest.h"

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"
#include "convergence_test.hpp"
#include "trace_analysis.hpp"
#include "validation_data.hpp"

#include <iostream>

struct probe_point {
    const char* label;
    nest::mc::segment_location where;
};

template <typename ProbePointSeq>
void run_ncomp_convergence_test(
    const char* model_name,
    const nest::mc::util::path& ref_data_path,
    nest::mc::backend_kind backend,
    const nest::mc::cell& c,
    ProbePointSeq& probe_points,
    float t_end=100.f)
{
    using namespace nest::mc;

    auto max_ncomp = g_trace_io.max_ncomp();
    auto dt = g_trace_io.min_dt();
    auto sample_dt = g_trace_io.sample_dt();

    nlohmann::json meta = {
        {"name", "membrane voltage"},
        {"model", model_name},
        {"dt", dt},
        {"sim", "nestmc"},
        {"units", "mV"},
        {"backend_kind", to_string(backend)}
    };

    auto exclude = stimulus_ends(c);

    auto n_probe = util::size(probe_points);
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

        hw::node_info nd(1, backend==backend_kind::gpu? 1: 0);
        auto decomp = partition_load_balance(rec, nd);
        model m(rec, decomp);

        runner.run(m, ncomp, sample_dt, t_end, dt, exclude);
    }
    runner.report();
    runner.assert_all_convergence();
}

void validate_ball_and_stick(nest::mc::backend_kind backend) {
    using namespace nest::mc;

    cell c = make_cell_ball_and_stick();
    probe_point points[] = {
        {"soma.mid", {0u, 0.5}},
        {"dend.mid", {1u, 0.5}},
        {"dend.end", {1u, 1.0}}
    };

    run_ncomp_convergence_test(
        "ball_and_stick",
        "neuron_ball_and_stick.json",
        backend,
        c,
        points);
}

void validate_ball_and_taper(nest::mc::backend_kind backend) {
    using namespace nest::mc;

    cell c = make_cell_ball_and_taper();
    probe_point points[] = {
        {"soma.mid",  {0u, 0.5}},
        {"taper.mid", {1u, 0.5}},
        {"taper.end", {1u, 1.0}}
    };

    run_ncomp_convergence_test(
        "ball_and_taper",
        "neuron_ball_and_taper.json",
        backend,
        c,
        points);
}

void validate_ball_and_3stick(nest::mc::backend_kind backend) {
    using namespace nest::mc;

    cell c = make_cell_ball_and_3stick();
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
        backend,
        c,
        points);
}

void validate_rallpack1(nest::mc::backend_kind backend) {
    using namespace nest::mc;

    cell c = make_cell_simple_cable();
    probe_point points[] = {
        {"cable.x0.0", {1u, 0.0}},
        {"cable.x0.3", {1u, 0.3}},
        {"cable.x1.0", {1u, 1.0}}
    };

    run_ncomp_convergence_test(
        "rallpack1",
        "numeric_rallpack1.json",
        backend,
        c,
        points,
        250.f);
}

void validate_ball_and_squiggle(nest::mc::backend_kind backend) {
    using namespace nest::mc;

    cell c = make_cell_ball_and_squiggle();
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
        backend,
        c,
        points);
}
