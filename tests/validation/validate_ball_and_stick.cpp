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

// TODO: further consolidate common code

TEST(ball_and_stick, neuron_ref) {
    // compare voltages against reference data produced from
    // nrn/ball_and_stick.py

    using namespace nlohmann;

    using lowered_cell = fvm::fvm_multicell<multicore::fvm_policy>;
    auto& V = g_trace_io;

    bool verbose = V.verbose();
    int max_ncomp = V.max_ncomp();

    // load validation data
    auto ref_data = V.load_traces("neuron_ball_and_stick.json");
    bool run_validation =
        ref_data.count("soma.mid") &&
        ref_data.count("dend.mid") &&
        ref_data.count("dend.end");

    EXPECT_TRUE(run_validation);

    // generate test data
    cell c = make_cell_ball_and_stick();
    add_common_voltage_probes(c);

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

        m.run(100, 0.001);

        for (auto& se: samplers) {
            std::string key = se.first;
            const simple_sampler& s = se.second;

            // save trace
            json meta = {
                {"name", "membrane voltage"},
                {"model", "ball_and_stick"},
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

TEST(ball_and_taper, neuron_ref) {
    // compare voltages against reference data produced from
    // nrn/ball_and_taper.py

    using namespace nlohmann;

    using lowered_cell = fvm::fvm_multicell<multicore::fvm_policy>;
    auto& V = g_trace_io;

    bool verbose = V.verbose();
    int max_ncomp = V.max_ncomp();

    // load validation data
    auto ref_data = V.load_traces("neuron_ball_and_taper.json");
    bool run_validation =
        ref_data.count("soma.mid") &&
        ref_data.count("taper.mid") &&
        ref_data.count("taper.end");

    EXPECT_TRUE(run_validation);

    // generate test data
    cell c = make_cell_ball_and_taper();
    add_common_voltage_probes(c);

    float sample_dt = .025;
    std::pair<const char *, simple_sampler> samplers[] = {
        {"soma.mid", simple_sampler(sample_dt)},
        {"taper.mid", simple_sampler(sample_dt)},
        {"taper.end", simple_sampler(sample_dt)}
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

        m.run(100, 0.001);

        for (auto& se: samplers) {
            std::string key = se.first;
            const simple_sampler& s = se.second;

            // save trace
            json meta = {
                {"name", "membrane voltage"},
                {"model", "ball_and_taper"},
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


TEST(ball_and_3stick, neuron_ref) {
    // compare voltages against reference data produced from
    // nrn/ball_and_3stick.py

    using namespace nlohmann;

    using lowered_cell = fvm::fvm_multicell<multicore::fvm_policy>;
    auto& V = g_trace_io;

    bool verbose = V.verbose();
    int max_ncomp = V.max_ncomp();

    // load validation data
    auto ref_data = V.load_traces("neuron_ball_and_3stick.json");
    bool run_validation =
        ref_data.count("soma.mid") &&
        ref_data.count("dend1.mid") &&
        ref_data.count("dend1.end") &&
        ref_data.count("dend2.mid") &&
        ref_data.count("dend2.end") &&
        ref_data.count("dend3.mid") &&
        ref_data.count("dend3.end");


    EXPECT_TRUE(run_validation);

    // generate test data
    cell c = make_cell_ball_and_3stick();
    add_common_voltage_probes(c);

    float sample_dt = .025;
    std::pair<const char *, simple_sampler> samplers[] = {
        {"soma.mid", simple_sampler(sample_dt)},
        {"dend1.mid", simple_sampler(sample_dt)},
        {"dend1.end", simple_sampler(sample_dt)},
        {"dend2.mid", simple_sampler(sample_dt)},
        {"dend2.end", simple_sampler(sample_dt)},
        {"dend3.mid", simple_sampler(sample_dt)},
        {"dend3.end", simple_sampler(sample_dt)}
    };

    std::map<std::string, std::vector<conv_entry<int>>> conv_results;

    for (int ncomp = 10; ncomp<max_ncomp; ncomp*=2) {
        for (auto& se: samplers) {
            se.second.reset();
        }
        c.cable(1)->set_compartments(ncomp);
        c.cable(2)->set_compartments(ncomp);
        c.cable(3)->set_compartments(ncomp);
        model<lowered_cell> m(singleton_recipe{c});

        // the ball-and-3stick-cell (should) have seven voltage probes:
        // centre of soma, followed by centre of section, end of section
        // for each of the three dendrite sections.

        for (unsigned i = 0; i < util::size(samplers); ++i) {
            m.attach_sampler({0u, i}, samplers[i].second.sampler<>());
        }

        m.run(100, 0.001);

        for (auto& se: samplers) {
            std::string key = se.first;
            const simple_sampler& s = se.second;

            // save trace
            json meta = {
                {"name", "membrane voltage"},
                {"model", "ball_and_3stick"},
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

