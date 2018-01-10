#include <fstream>
#include <utility>

#include <json/json.hpp>

#include <common_types.hpp>
#include <cell.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <simple_sampler.hpp>
#include <util/rangeutil.hpp>

#include "../gtest.h"

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"
#include "../test_util.hpp"

#include "trace_analysis.hpp"
#include "validation_data.hpp"

using namespace arb;

#if 0
// *Temporarily* disabled: compartment division policy
// will be moved to backend policy class.

/*
 * Expect dendtrites composed of a simple frustrum to give
 * essentially identical results no matter the compartment
 * division policy.
 */

template <typename CompPolicy>
std::vector<trace_data> run_model(const cell& c, float sample_dt, float t_end, float dt) {
    model<fvm::fvm_multicell<double, cell_local_size_type, div_compartment_by_ends>> m{singleton_recipe(c)};

    const auto& probes = m.probes();
    std::size_t n_probes = probes.size();
    std::vector<simple_sampler> samplers(n_probes, sample_dt);

    for (unsigned i = 0; i<n_probes; ++i) {
        m.attach_sampler(probes[i].id, samplers[i].sampler<>());
    }

    m.run(t_end, dt);
    std::vector<trace_data> traces;
    for (auto& s: samplers) {
        traces.push_back(std::move(s.trace));
    }
    return traces;
}


void run_test(cell&& c) {
    add_common_voltage_probes(c);

    float sample_dt = .025;
    float t_end = 100;
    float dt = 0.001;

    auto traces_by_ends = run_model<div_compartment_by_ends>(c, sample_dt, t_end, dt);
    auto traces_sampler = run_model<div_compartment_sampler>(c, sample_dt, t_end, dt);
    auto traces_integrator = run_model<div_compartment_integrator>(c, sample_dt, t_end, dt);

    auto n_trace = traces_by_ends.size();
    ASSERT_GT(n_trace, 0);
    ASSERT_EQ(n_trace, traces_sampler.size());
    ASSERT_EQ(n_trace, traces_integrator.size());

    for (unsigned i = 0; i<n_trace; ++i) {
        auto& t1 = traces_by_ends[i];
        auto& t2 = traces_sampler[i];
        auto& t3 = traces_integrator[i];

        // expect all traces to be (close to) the same
        double epsilon = 1e-6;
        double tol = epsilon*util::max_value(
            util::transform_view(values(t1), [](double x) { return std::abs(x); }));
        EXPECT_GE(tol, linf_distance(t1, t2));
        EXPECT_GE(tol, linf_distance(t2, t3));
        EXPECT_GE(tol, linf_distance(t3, t1));
    }
}

TEST(compartment_policy, validate_ball_and_stick) {
    SCOPED_TRACE("ball_and_stick");
    run_test(make_cell_ball_and_stick());
}

TEST(compartment_policy, validate_ball_and_3stick) {
    SCOPED_TRACE("ball_and_3stick");
    run_test(make_cell_ball_and_3stick());
}

TEST(compartment_policy, validate_ball_and_taper) {
    SCOPED_TRACE("ball_and_taper");
    run_test(make_cell_ball_and_taper());
}

#endif
