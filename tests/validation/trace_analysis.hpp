#pragma once

#include <vector>

#include "gtest.h"

#include <simple_sampler.hpp>
#include <math.hpp>
#include <util/optional.hpp>
#include <util/path.hpp>
#include <util/rangeutil.hpp>

namespace nest {
namespace mc {

/* Trace data comparison */

// Compute max |v_i - f(t_i)| where (t, v) is the 
// first trace `u` and f is the piece-wise linear interpolant
// of the second trace `ref`.

double linf_distance(const trace_data& u, const trace_data& ref);

// Find local maxima (peaks) in a trace, excluding end points.

struct trace_peak {
    float t;
    double v;
    float t_err;

    friend trace_peak operator-(trace_peak x, trace_peak y) {
        return {x.t-y.t, x.v-y.v, x.t_err+y.t_err};
    }
};

std::vector<trace_peak> local_maxima(const trace_data& u);

// Compare differences in peak times across two traces.
// Returns largest magnitute displacement between peaks,
// together with a sampling error bound, or `nothing`
// if the number of peaks differ.

util::optional<trace_peak> peak_delta(const trace_data& a, const trace_data& b);

// Record for error data for convergence testing.
// Only linf and peak_delta are used for convergence testing below;
// if and param are for record keeping in the validation test itself.

template <typename Param>
struct conv_entry {
    std::string id;
    Param param;
    double linf;
    util::optional<trace_peak> peak_delta;
};

template <typename Param>
using conv_data = std::vector<conv_entry<Param>>;

// Assert error convergence (gtest).

template <typename ConvEntrySeq>
void assert_convergence(const ConvEntrySeq& cs) {
    if (util::empty(cs)) return;

    auto tbound = [](trace_peak p) { return std::abs(p.t)+p.t_err; };
    float peak_dt_bound = math::infinity<>();

    for (auto pi = std::begin(cs); std::next(pi)!=std::end(cs); ++pi) {
        const auto& p = *pi;
        const auto& c = *std::next(pi);

        EXPECT_LE(c.linf, p.linf) << "L∞ error increase";

        if (!c.peak_delta) {
            EXPECT_FALSE(p.peak_delta) << "divergence in peak count";
        }
        else {
            double t = std::abs(c.peak_delta->t);
            double t_limit = c.peak_delta->t_err+peak_dt_bound;

            EXPECT_LE(t, t_limit) << "divergence in max peak displacement";

            peak_dt_bound = std::min(peak_dt_bound, tbound(*c.peak_delta));
        }
    }
}

// Report table of convergence results.

template <typename ConvEntrySeq>
void report_conv_table(std::ostream& out, const ConvEntrySeq& tbl, const std::string& param_name) {
    out << "id," << param_name << ",linf,peak_dt,peak_dt_err\n";
    for (const auto& c: tbl) {
        out << c.id << "," << c.param << "," << c.linf << ",";
        if (c.peak_delta) {
            out << c.peak_delta->t << "," << c.peak_delta->t_err << "\n";
        }
        else {
            out << "NA,NA\n";
        }
    }
}

} // namespace mc
} // namespace nest
