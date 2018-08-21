#include <algorithm>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "../gtest.h"

#include <arbor/math.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/util/optional.hpp>

#include "interpolate.hpp"
#include "trace_analysis.hpp"

namespace arb {

struct trace_interpolant {
    trace_interpolant(const trace_data<double>& trace): trace_(trace) {}

    double operator()(float t) const {
        return pw_linear_interpolate(t, trace_,
            [](auto& entry) { return entry.t; },
            [](auto& entry) { return entry.v; });
    }

    const trace_data<double>& trace_;
};

double linf_distance(const trace_data<double>& u, const trace_data<double>& r) {
    trace_interpolant f{r};

    double linf = 0;
    for (auto entry: u) {
        linf = std::max(linf, std::abs(entry.v-f(entry.t)));
    }

    return linf;
}

// Compute linf distance as above, but excluding sample points that lie
// near points in `excl`.
//
// `excl` contains the times to exclude, in ascending order.

double linf_distance(const trace_data<double>& u, const trace_data<double>& ref, const std::vector<float>& excl) {
    trace_interpolant f{ref};

    trace_data<double> reduced;
    unsigned nexcl = excl.size();
    unsigned ei = 0;

    unsigned nu = u.size();
    unsigned ui = 0;

    while (ei<nexcl && ui<nu) {
        float t = excl[ei++];

        unsigned uj = ui;
        while (uj<nu && u[uj].t<t) ++uj;

        // include points up to and including uj-2, and then proceed from point uj+1,
        // excluding the two points closest to the discontinuity.

        for (unsigned k = ui; k+1<uj; ++k) {
            reduced.push_back(u[k]);
        }
        ui = uj+1;
    }

    for (auto k = ui; k<nu; ++k) {
        reduced.push_back(u[k]);
    }

    return linf_distance(reduced, ref);
}

std::vector<trace_peak> local_maxima(const trace_data<double>& u) {
    std::vector<trace_peak> peaks;
    if (u.size()<2) return peaks;

    int s_prev = math::signum(u[1].v-u[0].v);
    std::size_t i_start = 0;

    for (std::size_t i = 2; i<u.size()-1; ++i) {
        int s = math::signum(u[i].v-u[i-1].v);
        if (s_prev==1 && s==-1) {
            // found peak between i_start and i,
            // observerd peak value at i-1.
            float t0 = u[i_start].t;
            float t1 = u[i].t;

            peaks.push_back({(t0+t1)/2, u[i-1].v, (t1-t0)/2});
        }

        if (s!=0) {
            s_prev = s;
            if (s_prev>0) {
                i_start = i-1;
            }
        }
    }
    return peaks;
}

util::optional<trace_peak> peak_delta(const trace_data<double>& a, const trace_data<double>& b) {
    auto p = local_maxima(a);
    auto q = local_maxima(b);

    if (p.size()!=q.size() || p.empty()) return util::nullopt;

    auto max_delta = p[0]-q[0];

    for (std::size_t i = 1u; i<p.size(); ++i) {
        auto delta = p[i]-q[i];
        // pick maximum delta by greatest lower bound on dt
        if (std::abs(delta.t)-delta.t_err>std::abs(max_delta.t)-max_delta.t_err) {
            max_delta = delta;
        }
    }
    return max_delta;
}

} // namespace arb

