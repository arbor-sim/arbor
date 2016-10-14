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
#include <util/transform.hpp>

#include "gtest.h"

#include "../test_common_cells.hpp"
#include "../test_util.hpp"
#include "trace_analysis.hpp"
#include "validation_data.hpp"

using namespace nest::mc;

void run_synapse_test(const char* syn_type, const char* ref_file) {
    using namespace nlohmann;

    using lowered_cell = fvm::fvm_multicell<double, cell_local_size_type>;
    auto& V = g_trace_io;

    bool verbose = V.verbose();
    int max_ncomp = V.max_ncomp();

    // load validation data
    auto ref_data = V.load_traces(ref_file);
    bool run_validation =
        ref_data.count("soma.mid") &&
        ref_data.count("dend.mid") &&
        ref_data.count("dend.end");

    EXPECT_TRUE(run_validation);

    // generate test data
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

    float sample_dt = .025;
    std::pair<const char *, simple_sampler> samplers[] = {
        {"soma.mid", simple_sampler(sample_dt)},
        {"dend.mid", simple_sampler(sample_dt)},
        {"dend.end", simple_sampler(sample_dt)}
    };

    std::map<std::string, std::vector<conv_entry<int>>> conv_results;

    for (int ncomp = 10; ncomp<max_ncomp; ncomp*=2) {
        for (auto& se: samplers) {
            se.second.reset();
        }
        c.cable(1)->set_compartments(ncomp);
        model<lowered_cell> m(singleton_recipe{c});

        // the ball-and-stick-cell (should) have three voltage probes:
        // centre of soma, centre of dendrite, end of dendrite.

        m.attach_sampler({0u, 0u}, samplers[0].second.sampler<>());
        m.attach_sampler({0u, 1u}, samplers[1].second.sampler<>());
        m.attach_sampler({0u, 2u}, samplers[2].second.sampler<>());

        m.group(0).enqueue_events(synthetic_events);
        m.run(70, 0.001);

        for (auto& se: samplers) {
            std::string key = se.first;
            const simple_sampler& s = se.second;

            // save trace
            json meta = {
                {"name", "membrane voltage"},
                {"model", syn_type},
                {"sim", "nestmc"},
                {"ncomp", ncomp},
                {"units", "mV"}};

            V.save_trace(key, s.trace, meta);

            // compute metrics
            if (run_validation) {
                double linf = linf_distance(s.trace, ref_data[key]);
                auto pd = peak_delta(s.trace, ref_data[key]);

                conv_results[key].push_back({key, ncomp, linf, pd});
            }
        }
    }

    if (verbose && run_validation) {
        report_conv_table(std::cout, conv_results, "ncomp");
    }

    for (auto key: util::transform_view(samplers, util::first)) {
        SCOPED_TRACE(key);

        const auto& results = conv_results[key];
        assert_convergence(results);
    }
}

TEST(simple_synapse, expsyn_neuron_ref) {
    SCOPED_TRACE("expsyn");
    run_synapse_test("expsyn", "neuron_simple_exp_synapse.json");
}

TEST(simple_synapse, exp2syn_neuron_ref) {
    SCOPED_TRACE("exp2syn");
    run_synapse_test("exp2syn", "neuron_simple_exp2_synapse.json");
}

