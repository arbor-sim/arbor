#pragma once

#include <vector>

#include <model.hpp>
#include <sampling.hpp>
#include <simple_sampler.hpp>
#include <util/filter.hpp>
#include <util/rangeutil.hpp>

#include <json/json.hpp>

#include "../gtest.h"

#include "trace_analysis.hpp"
#include "validation_data.hpp"

namespace nest {
namespace mc {

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
        util::assign(probe_labels_, probe_labels);
    }

    // Allow free access to JSON meta data attached to saved traces.
    nlohmann::json& metadata() { return meta_; }

    void load_reference_data(const util::path& ref_path) {
        run_validation_ = false;
        try {
            ref_data_ = g_trace_io.load_traces(ref_path);

            run_validation_ = util::all_of(probe_labels_,
                [&](const probe_label& pl) { return ref_data_.count(pl.label)>0; });

            EXPECT_TRUE(run_validation_);
        }
        catch (std::runtime_error&) {
            ADD_FAILURE() << "failure loading reference data: " << ref_path;
        }
    }

    void run(model& m, Param p, float sample_dt, float t_end, float dt, const std::vector<float>& excl) {
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
            entry.h = m.add_sampler(one_probe(pl.probe_id), regular_schedule(sample_dt), simple_sampler<double>(entry.trace));
        }

        m.run(t_end, dt);

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
            m.remove_sampler(entry.h);
        }
    }

    void report() {
        if (run_validation_ && g_trace_io.verbose()) {
            // reorder to group by id
            util::stable_sort_by(conv_tbl_, [](const conv_entry<Param>& e) { return e.id; });
            report_conv_table(std::cout, conv_tbl_, param_name_);
        }
    }

    void assert_all_convergence() const {
        for (const auto& pl: probe_labels_) {
            SCOPED_TRACE(pl.label);
            assert_convergence(util::filter(conv_tbl_,
                        [&](const conv_entry<Param>& e) { return e.id==pl.label; }));
        }
    }
};

/*
 * Extract time points to exclude from current stimulus end-points.
 */

inline std::vector<float> stimulus_ends(const cell& c) {
    std::vector<float> ts;

    for (const auto& stimulus: c.stimuli()) {
        float t0 = stimulus.clamp.delay();
        float t1 = t0+stimulus.clamp.duration();
        ts.push_back(t0);
        ts.push_back(t1);
    }

    util::sort(ts);
    return ts;
}

} // namespace mc
} // namespace nest
