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

struct sampler_info {
    const char* label;
    cell_member_type probe;
    simple_sampler sampler;
};

template <
    typename lowered_cell,
    typename SamplerInfoSeq
>
void run_ncomp_convergence_test(
    const char* model_name,
    const char* refdata_path,
    const cell& c,
    SamplerInfoSeq& samplers,
    float t_end=100.f)
{
    using nlohmann::json;

    SCOPED_TRACE(model_name);

    auto& V = g_trace_io;
    bool verbose = V.verbose();
    int max_ncomp = V.max_ncomp();

    auto keys = util::transform_view(samplers,
        [](const sampler_info& se) { return se.label; });

    bool run_validation = false;
    std::map<std::string, trace_data> ref_data;
    try {
        ref_data = V.load_traces(refdata_path);

        run_validation = std::all_of(keys.begin(), keys.end(),
            [&](const char* key) { return ref_data.count(key)>0; });

        EXPECT_TRUE(run_validation);
    }
    catch (std::runtime_error&) {
        ADD_FAILURE() << "failure loading reference data: " << refdata_path;
    }

    std::map<std::string, std::vector<conv_entry<int>>> conv_results;

    for (int ncomp = 10; ncomp<max_ncomp; ncomp*=2) {
        for (auto& seg: c.segments()) {
            if (!seg->is_soma()) {
                seg->set_compartments(ncomp);
            }
        }
        model<lowered_cell> m(singleton_recipe{c});

        // reset samplers and attach to probe locations
        for (auto& se: samplers) {
            se.sampler.reset();
            m.attach_sampler(se.probe, se.sampler.template sampler<>());
        }

        m.run(100, 0.001);

        for (auto& se: samplers) {
            std::string key = se.label;
            const simple_sampler& s = se.sampler;

            // save trace
            json meta = {
                {"name", "membrane voltage"},
                {"model", model_name},
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

    for (auto key: keys) {
        SCOPED_TRACE(key);

        const auto& results = conv_results[key];
        assert_convergence(results);
    }
}

TEST(ball_and_stick, neuron_ref) {
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

TEST(ball_and_taper, neuron_ref) {
    using lowered_cell = fvm::fvm_multicell<double, cell_local_size_type>;

    cell c = make_cell_ball_and_taper();
    add_common_voltage_probes(c);

    float sample_dt = 0.025f;
    sampler_info samplers[] = {
        {"soma.mid", {0u, 0u}, simple_sampler(sample_dt)},
        {"taper.mid", {0u, 1u}, simple_sampler(sample_dt)},
        {"taper.end", {0u, 2u}, simple_sampler(sample_dt)}
    };

    run_ncomp_convergence_test<lowered_cell>(
        "ball_and_taper",
        "neuron_ball_and_taper.json",
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
        samplers);
}

#if 0
TEST(ball_and_stick, neuron_ref) {
    // compare voltages against reference data produced from
    // nrn/ball_and_stick.py

    using namespace nlohmann;

    using lowered_cell = fvm::fvm_multicell<double, cell_local_size_type>;
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

    using lowered_cell = fvm::fvm_multicell<double, cell_local_size_type>;
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

    using lowered_cell = fvm::fvm_multicell<double, cell_local_size_type>;
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
#endif

