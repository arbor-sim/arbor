#pragma once

#include <vector>

#include "gtest.h"

#include <simple_sampler.hpp>
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

template <typename Param>
struct conv_entry {
    std::string id;
    Param param;
    double linf;
    util::optional<trace_peak> pd;
};

template <typename Param>
using conv_data = std::vector<conv_entry<Param>>;

// Assert error convergence (gtest).

template <typename Param>
void assert_convergence(const conv_data<Param>& cs) {
    if (cs.size()<2) return;

    auto tbound = [](trace_peak p) { return std::abs(p.t)+p.t_err; };
    auto smallest_pd = cs[0].pd;

    for (unsigned i = 1; i<cs.size(); ++i) {
        const auto& p = cs[i-1];
        const auto& c = cs[i];

        EXPECT_LE(c.linf, p.linf) << "L∞ error increase";
        EXPECT_TRUE(c.pd || (!c.pd && !p.pd)) << "divergence in peak count";

        if (c.pd && smallest_pd) {
            double t = std::abs(c.pd->t);
            EXPECT_LE(t, c.pd->t_err+tbound(*smallest_pd)) << "divergence in max peak displacement";
        }

        if (c.pd && (!smallest_pd || tbound(*c.pd)<tbound(*smallest_pd))) {
            smallest_pd = c.pd;
        }
    }
}

// Report table of convergence results.
// (Takes collection with pair<string, conv_data>
// entries.)

template <typename Map>
void report_conv_table(std::ostream& out, const Map& tbl, const std::string& param_name) {
    out << "location," << param_name << ",linf,peak_dt,peak_dt_err\n";
    for (const auto& p: tbl) {
        const auto& location = p.first;
        for (const auto& c: p.second) {
            out << location << "," << c.param << "," << c.linf << ",";
            if (c.pd) {
                out << c.pd->t << "," << c.pd->t_err << "\n";
            }
            else {
                out << "NA,NA\n";
            }
        }
    }
}

} // namespace mc
} // namespace nest
