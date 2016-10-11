#pragma once

#include <util/filter.hpp>
#include <util/rangeutil.hpp>

#include <json/json.hpp>

#include "gtest.h"

#include "trace_analysis.hpp"
#include "validation_data.hpp"

namespace nest {
namespace mc {

struct sampler_info {
    const char* label;
    cell_member_type probe;
    simple_sampler sampler;
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
    std::vector<sampler_info> cell_samplers_;
    std::map<std::string, trace_data> ref_data_;
    std::vector<conv_entry<Param>> conv_tbl_;

public:
    template <typename SamplerInfoSeq>
    convergence_test_runner(
        const std::string& param_name,
        const SamplerInfoSeq& samplers,
        const nlohmann::json meta
    ):
        param_name_(param_name),
        run_validation_(false),
        meta_(meta)
    {
        util::assign(cell_samplers_, samplers);
    }

    // allow free access to JSON meta data attached to saved traces
    nlohmann::json& metadata() { return meta_; }

    void load_reference_data(const util::path& ref_path) {
        run_validation_ = false;
        try {
            ref_data_ = g_trace_io.load_traces(ref_path);

            run_validation_ = util::all_of(cell_samplers_,
                [&](const sampler_info& se) { return ref_data_.count(se.label)>0; });

            EXPECT_TRUE(run_validation_);
        }
        catch (std::runtime_error&) {
            ADD_FAILURE() << "failure loading reference data: " << ref_path;
        }
    }

    template <typename Model>
    void run(Model& m, Param p, float t_end, float dt, const std::vector<float>& excl={}) {
        // reset samplers and attach to probe locations
        for (auto& se: cell_samplers_) {
            se.sampler.reset();
            m.attach_sampler(se.probe, se.sampler.template sampler<>());
        }

        m.run(t_end, dt);

        for (auto& se: cell_samplers_) {
            std::string label = se.label;
            const auto& trace = se.sampler.trace;

            // save trace
            nlohmann::json trace_meta{meta_};
            trace_meta[param_name_] = p;

            g_trace_io.save_trace(label, trace, trace_meta);

            // compute metrics
            if (run_validation_) {
                double linf = linf_distance(trace, ref_data_[label], excl);
                auto pd = peak_delta(trace, ref_data_[label]);

                conv_tbl_.push_back({label, p, linf, pd});
            }
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
        for (const sampler_info& se: cell_samplers_) {
            SCOPED_TRACE(se.label);
            assert_convergence(util::filter(conv_tbl_,
                        [&](const conv_entry<Param>& e) { return e.id==se.label; }));
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
