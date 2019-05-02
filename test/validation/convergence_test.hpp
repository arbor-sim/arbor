#pragma once

#include <iterator>
#include <vector>

#include <nlohmann/json.hpp>

#include <arbor/sampling.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/schedule.hpp>
#include <sup/path.hpp>

#include "../gtest.h"

#include "trace_analysis.hpp"
#include "validation_data.hpp"

namespace arb {

struct probe_label {
    const char* label;
    cell_member_type probe_id;
};

/* Common functionality for testing convergence towards
 * a reference data set as some parameter of the model
 * is changed.
 *
 * Type parameter Param is the type of the parameter that
 * is changed between runs.
 */

template <typename Param>
class convergence_test_runner {
private:
    std::string param_name_;
    bool run_validation_;
    nlohmann::json meta_;
    std::vector<probe_label> probe_labels_;
    std::map<std::string, trace_data<double>> ref_data_;
    std::vector<conv_entry<Param>> conv_tbl_;

public:
    template <typename ProbeLabelSeq>
    convergence_test_runner(
        const std::string& param_name,
        const ProbeLabelSeq& probe_labels,
        const nlohmann::json& meta
    ):
        param_name_(param_name),
        run_validation_(false),
        meta_(meta)
    {
        using std::begin;
        using std::end;

        probe_labels_.assign(begin(probe_labels), end(probe_labels));
    }

    // Allow free access to JSON meta data attached to saved traces.
    nlohmann::json& metadata() { return meta_; }

    void load_reference_data(const sup::path& ref_path) {
        run_validation_ = false;
        try {
            ref_data_ = g_trace_io.load_traces(ref_path);

            run_validation_ = true;
            for (const auto& pl: probe_labels_) {
                if (!(ref_data_.count(pl.label)>0)) {
                    run_validation_ = false;
                    break;
                }
            }

            EXPECT_TRUE(run_validation_);
        }
        catch (std::runtime_error&) {
            ADD_FAILURE() << "failure loading reference data: " << ref_path;
        }
    }

    void run(simulation& sim, Param p, float sample_dt, float t_end, float dt, const std::vector<float>& excl) {
        struct sampler_state {
            sampler_association_handle h; // Keep these for clean up at end.
            const char* label;
            trace_data<double> trace;
        };

        // Attach new samplers to labelled probe ids.
        std::vector<sampler_state> samplers;
        samplers.reserve(probe_labels_.size());

        for (auto& pl: probe_labels_) {
            samplers.push_back({});
            auto& entry = samplers.back();

            entry.label = pl.label;
            entry.h = sim.add_sampler(one_probe(pl.probe_id), regular_schedule(sample_dt), simple_sampler<double>(entry.trace));
        }

        sim.run(t_end, dt);

        for (auto& entry: samplers) {
            std::string label = entry.label;
            const auto& trace = entry.trace;

            // Save trace.
            nlohmann::json trace_meta(meta_);
            trace_meta[param_name_] = p;

            g_trace_io.save_trace(label, trace, trace_meta);

            // Compute metreics.
            if (run_validation_) {
                double linf = linf_distance(ref_data_[label], trace, excl);
                auto pd = peak_delta(trace, ref_data_[label]);

                conv_tbl_.push_back({label, p, linf, pd});
            }
        }

        // Remove added samplers.
        for (const auto& entry: samplers) {
            sim.remove_sampler(entry.h);
        }
    }

    void report() {
        if (run_validation_ && g_trace_io.verbose()) {
            // reorder to group by id
            std::stable_sort(conv_tbl_.begin(), conv_tbl_.end(),
                [](const auto& a, const auto& b) { return a.id<b.id; });

            report_conv_table(std::cout, conv_tbl_, param_name_);
        }
    }

    void assert_all_convergence() const {
        std::vector<conv_entry<Param>> with_label;

        for (const auto& pl: probe_labels_) {
            SCOPED_TRACE(pl.label);

            with_label.clear();
            for (const auto& e: conv_tbl_) {
                if (e.id==pl.label) {
                    with_label.push_back(e);
                }
            }

            assert_convergence(with_label);
        }
    }
};

/*
 * Extract time points to exclude from current stimulus end-points.
 */

inline std::vector<float> stimulus_ends(const cable_cell& c) {
    std::vector<float> ts;

    for (const auto& stimulus: c.stimuli()) {
        float t0 = stimulus.clamp.delay;
        float t1 = t0+stimulus.clamp.duration;
        ts.push_back(t0);
        ts.push_back(t1);
    }

    std::sort(ts.begin(), ts.end());
    return ts;
}

} // namespace arb
