#include <cfloat>
#include <cmath>
#include <numeric>
#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/morph/locset.hpp>

#include "morph/em_morphology.hpp"
#include "util/maputil.hpp"

namespace arb {

void check_global_properties(const cable_cell_global_properties& G) {
    auto& param = G.default_parameters;

    if (!param.init_membrane_potential) {
        throw cable_cell_error("missing global default parameter value: init_membrane_potential");
    }

    if (!param.temperature_K) {
        throw cable_cell_error("missing global default parameter value: temperature");
    }

    if (!param.axial_resistivity) {
        throw cable_cell_error("missing global default parameter value: axial_resistivity");
    }

    if (!param.membrane_capacitance) {
        throw cable_cell_error("missing global default parameter value: membrane_capacitance");
    }

    for (const auto& ion: util::keys(G.ion_species)) {
        if (!param.ion_data.count(ion)) {
            throw cable_cell_error("missing ion defaults for ion "+ion);
        }
    }

    for (const auto& kv: param.ion_data) {
        auto& ion = kv.first;
        const cable_cell_ion_data& data = kv.second;
        if (std::isnan(data.init_int_concentration)) {
            throw cable_cell_error("missing init_int_concentration for ion "+ion);
        }
        if (std::isnan(data.init_ext_concentration)) {
            throw cable_cell_error("missing init_ext_concentration for ion "+ion);
        }
        if (std::isnan(data.init_reversal_potential) && !param.reversal_potential_method.count(ion)) {
            throw cable_cell_error("missing init_reversal_potential or reversal_potential_method for ion "+ion);
        }
    }
}

cable_cell_local_parameter_set neuron_parameter_defaults = {
    // ion defaults: internal concentration [mM], external concentration [mM], reversal potential [mV]
    {
        {"na", {10.0,  140.0,  115 - 65.}},
        {"k",  {54.4,    2.5,  -12 - 65.}},
        {"ca", {5e-5,    2.0,  12.5*std::log(2.0/5e-5)}}
    },
    // initial membrane potential [mV]
    -65.0,
    // temperatue [K]
    6.3 + 273.15,
    // axial resistivity [Ω·cm]
    35.4,
    // membrane capacitance [F/m²]
    0.01
};

// Discretization policy implementations:

locset cv_policy_max_extent::cv_boundary_points(const cable_cell& cell) const {
    const auto& emorph = *cell.morphology();
    const unsigned nbranch = emorph.morph().num_branches();
    if (!nbranch || max_extent_<=0) return ls::nil();

    std::vector<mlocation> points;

    unsigned bidx = 0;
    if (flags_&cv_policy_flag::single_root_cv) {
        points.push_back({0, 0.});
        points.push_back({0, 1.});
        bidx = 1;
    }

    const double oomax_extent = 1./max_extent_;
    while (bidx<nbranch) {
        unsigned ncv = std::ceil(emorph.branch_length(bidx)*oomax_extent);
        double ooncv = 1./ncv;

        if (flags_&cv_policy_flag::interior_forks) {
            for (unsigned i = 0; i<ncv; ++i) {
                points.push_back({bidx, (1+2*i)*ooncv/2});
            }
        }
        else {
            for (unsigned i = 0; i<ncv; ++i) {
                points.push_back({bidx, i*ooncv});
            }
            points.push_back({bidx, 1.});
        }
        ++bidx;
    }

    return std::accumulate(points.begin(), points.end(), ls::nil(), [](auto& l, auto& p) { return join(l, ls::location(p)); });
}

locset cv_policy_fixed_per_branch::cv_boundary_points(const cable_cell& cell) const {
    const unsigned nbranch = cell.morphology()->morph().num_branches();
    if (!nbranch) return ls::nil();

    std::vector<mlocation> points;

    unsigned bidx = 0;
    if (flags_&cv_policy_flag::single_root_cv) {
        points.push_back({0, 0.});
        points.push_back({0, 1.});
        bidx = 1;
    }

    double ooncv = 1./cv_per_branch_;
    while (bidx<nbranch) {
        if (flags_&cv_policy_flag::interior_forks) {
            for (unsigned i = 0; i<cv_per_branch_; ++i) {
                points.push_back({bidx, (1+2*i)*ooncv/2});
            }
        }
        else {
            for (unsigned i = 0; i<cv_per_branch_; ++i) {
                points.push_back({bidx, i*ooncv});
            }
            points.push_back({bidx, 1.});
        }
        ++bidx;
    }

    return std::accumulate(points.begin(), points.end(), ls::nil(), [](auto& l, auto& p) { return join(l, ls::location(p)); });
}

} // namespace arb
