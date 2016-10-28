#include <fstream>
#include <utility>

#include <json/json.hpp>

#include <common_types.hpp>
#include <cell.hpp>
#include <fvm_multicell.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <simple_sampler.hpp>
#include <util/rangeutil.hpp>

#include "gtest.h"

#include "../test_common_cells.hpp"
#include "convergence_test.hpp"
#include "trace_analysis.hpp"
#include "validation_data.hpp"

using namespace nest::mc;

TEST(soma, numeric_ref) {
    using lowered_cell = fvm::fvm_multicell<double, cell_local_size_type>;

    cell c = make_cell_soma_only();
    add_common_voltage_probes(c);
    model<lowered_cell> m(singleton_recipe{c});

    float sample_dt = .025f;
    sampler_info samplers[] = {{"soma.mid", {0u, 0u}, simple_sampler(sample_dt)}};

    nlohmann::json meta = {
        {"name", "membrane voltage"},
        {"model", "soma"},
        {"sim", "nestmc"},
        {"units", "mV"}
    };

    convergence_test_runner<float> R("dt", samplers, meta);
    R.load_reference_data("numeric_soma.json");

    float t_end = 100.f;
    for (auto dt: {0.05f, 0.02f, 0.01f, 0.005f, 0.001f}) {
        m.reset();
        R.run(m, dt, t_end, dt);
    }
    R.report();
    R.assert_all_convergence();
}
