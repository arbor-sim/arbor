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

#include "../test_util.hpp"
#include "../test_common_cells.hpp"
#include "trace_analysis.hpp"
#include "validation_data.hpp"

using namespace nest::mc;

TEST(soma, numeric_ref) {
    // compare voltages against reference data produced from
    // nrn/ball_and_taper.py

    using namespace nlohmann;

    using lowered_cell = fvm::fvm_multicell<multicore::fvm_policy>;
    auto& V = g_trace_io;

    bool verbose = V.verbose();

    // load validation data

    bool run_validation = false;
    std::map<std::string, trace_data> ref_data;
    const char* key = "soma.mid";

    const char* ref_data_path = "numeric_soma.json";
    try {
        ref_data = V.load_traces(ref_data_path);
        run_validation = ref_data.count(key);

        EXPECT_TRUE(run_validation);
    }
    catch (std::runtime_error&) {
        ADD_FAILURE() << "failure loading reference data: " << ref_data_path;
    }

    // generate test data
    cell c = make_cell_soma_only();
    add_common_voltage_probes(c);

    float sample_dt = .025;
    simple_sampler sampler(sample_dt);
    conv_data<float> convs;

    for (auto dt: {0.05f, 0.02f, 0.01f, 0.005f, 0.001f}) {
        sampler.reset();
        model<lowered_cell> m(singleton_recipe{c});

        m.attach_sampler({0u, 0u}, sampler.sampler<>());
        m.run(100, dt);

        // save trace
        auto& trace = sampler.trace;
        json meta = {
            {"name", "membrane voltage"},
            {"model", "soma"},
            {"sim", "nestmc"},
            {"dt", dt},
            {"units", "mV"}};

        V.save_trace(key, trace, meta);

        // compute metrics
        if (run_validation) {
            double linf = linf_distance(trace, ref_data[key]);
            auto pd = peak_delta(trace, ref_data[key]);

            convs.push_back({key, dt, linf, pd});
        }
    }

    if (verbose && run_validation) {
        std::map<std::string, std::vector<conv_entry<float>>> conv_results = {{key, convs}};
        report_conv_table(std::cout, conv_results, "dt");
    }

    SCOPED_TRACE("soma.mid");
    assert_convergence(convs);
}
