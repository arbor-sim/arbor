#include <cfloat>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>
#include <variant>
#include <tuple>

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/s_expr.hpp>

#include "util/maputil.hpp"

namespace arb {

ARB_ARBOR_API void check_global_properties(const cable_cell_global_properties& G) {
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

    for (const auto& [ion, data]: param.ion_data) {
        if (!data.init_int_concentration) {
            throw cable_cell_error("missing init_int_concentration for ion "+ion);
        }
        if (!data.init_ext_concentration) {
            throw cable_cell_error("missing init_ext_concentration for ion "+ion);
        }
        if (data.diffusivity && *data.diffusivity < 0.0) {
            throw cable_cell_error("negative diffusivity for ion "+ion);
        }
        if (!data.init_reversal_potential && !param.reversal_potential_method.count(ion)) {
            throw cable_cell_error("missing init_reversal_potential or reversal_potential_method for ion "+ion);
        }
    }
}

cable_cell_parameter_set neuron_parameter_defaults = {
    // initial membrane potential [mV]
    -65.0,
    // temperatue [K]
    6.3 + 273.15,
    // axial resistivity [Ω·cm]
    35.4,
    // membrane capacitance [F/m²]
    0.01,
    // ion defaults:
    // internal concentration [mM], external concentration [mM], reversal potential [mV], diffusivity [m^2/s]
    {{"na", {10.0,  140.0,  115 - 65.,               0.0}},
     {"k",  {54.4,    2.5,  -12 - 65.,               0.0}},
     {"ca", {5e-5,    2.0,  12.5*std::log(2.0/5e-5), 0.0}}
    },
};


std::vector<defaultable> cable_cell_parameter_set::serialize() const {
    std::vector<defaultable> D;
    if (init_membrane_potential) {
        D.push_back(arb::init_membrane_potential{*this->init_membrane_potential});
    }
    if (temperature_K) {
        D.push_back(arb::temperature_K{*this->temperature_K});
    }
    if (axial_resistivity) {
        D.push_back(arb::axial_resistivity{*this->axial_resistivity});
    }
    if (membrane_capacitance) {
        D.push_back(arb::membrane_capacitance{*this->membrane_capacitance});
    }

    for (const auto& [name, data]: ion_data) {
        if (data.init_int_concentration) {
            D.push_back(init_int_concentration{name, *data.init_int_concentration});
        }
        if (data.init_ext_concentration) {
            D.push_back(init_ext_concentration{name, *data.init_ext_concentration});
        }
        if (data.init_reversal_potential) {
            D.push_back(init_reversal_potential{name, *data.init_reversal_potential});
        }
        if (data.diffusivity) {
            D.push_back(ion_diffusivity{name, *data.diffusivity});
        }
    }

    for (const auto& [name, mech]: reversal_potential_method) {
        D.push_back(ion_reversal_potential_method{name, mech});
    }

    if (discretization) {
        D.push_back(*discretization);
    }

    return D;
}

decor& decor::paint(region where, paintable what) {
    paintings_.emplace_back(std::move(where), std::move(what));
    return *this;
}

decor& decor::place(locset where, placeable what, cell_tag_type label) {
    placements_.emplace_back(std::move(where), std::move(what), std::move(label));
    return *this;
}

decor& decor::set_default(defaultable what) {
    // NOTE: the index/get approach is considerably faster than std::visit.
    switch (what.index()) {
        case 0: {
            auto& p = std::get<init_membrane_potential>(what);
            if (p.value.type() != iexpr_type::scalar) throw cable_cell_error{"Default values cannot have a scale."};
            defaults_.init_membrane_potential = *p.value.get_scalar();
            break;
        }
        case 1: {
            auto& p = std::get<axial_resistivity>(what);
            if (p.value.type() != iexpr_type::scalar) throw cable_cell_error{"Default values cannot have a scale."};
            defaults_.axial_resistivity = *p.value.get_scalar();
            break;
        }
        case 2: {
            auto& p = std::get<temperature_K>(what);
            if (p.value.type() != iexpr_type::scalar) throw cable_cell_error{"Default values cannot have a scale."};
            defaults_.temperature_K = *p.value.get_scalar();
            break;
        }
        case 3: {
            auto& p = std::get<membrane_capacitance>(what);
            if (p.value.type() != iexpr_type::scalar) throw cable_cell_error{"Default values cannot have a scale."};
            defaults_.membrane_capacitance = *p.value.get_scalar();
            break;
        }
        case 4: {
            auto& p = std::get<ion_diffusivity>(what);
            if (p.value.type() != iexpr_type::scalar) throw cable_cell_error{"Default values cannot have a scale."};
            defaults_.ion_data[p.ion].diffusivity = p.value.get_scalar();
            break;
        }
        case 5: {
            auto& p = std::get<init_int_concentration>(what);
            if (p.value.type() != iexpr_type::scalar) throw cable_cell_error{"Default values cannot have a scale."};
            defaults_.ion_data[p.ion].init_int_concentration = *p.value.get_scalar();
            break;
        }
        case 6: {
            auto& p = std::get<init_ext_concentration>(what);
            if (p.value.type() != iexpr_type::scalar) throw cable_cell_error{"Default values cannot have a scale."};
            defaults_.ion_data[p.ion].init_ext_concentration = p.value.get_scalar();
            break;
        }
        case 7: {
            auto& p = std::get<init_reversal_potential>(what);
            if (p.value.type() != iexpr_type::scalar) throw cable_cell_error{"Default values cannot have a scale."};
            defaults_.ion_data[p.ion].init_reversal_potential = *p.value.get_scalar();
            break;
        }
        case 8: {
            auto& p = std::get<ion_reversal_potential_method>(what);
            defaults_.reversal_potential_method[p.ion] = p.method;
            break;
        }
        case 9:
            defaults_.discretization = std::forward<cv_policy>(std::get<cv_policy>(what));
            break;
        default:
            throw arbor_internal_error{"Unknown defaultable variant"};
    }
    return *this;
}

} // namespace arb
