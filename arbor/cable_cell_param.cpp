#include <cfloat>
#include <cmath>
#include <numeric>
#include <vector>
#include <variant>

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>

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
        if (!data.init_int_concentration) {
            throw cable_cell_error("missing init_int_concentration for ion "+ion);
        }
        if (!data.init_ext_concentration) {
            throw cable_cell_error("missing init_ext_concentration for ion "+ion);
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
    // internal concentration [mM], external concentration [mM], reversal potential [mV]
    {
        {"na", {10.0,  140.0,  115 - 65.}},
        {"k",  {54.4,    2.5,  -12 - 65.}},
        {"ca", {5e-5,    2.0,  12.5*std::log(2.0/5e-5)}}
    },
};


// s-expression printers for paintable and placeable items.

std::ostream& sstring(std::ostream& o, const mechanism_desc& d) {
    o << "(mechanism \"" << d.name() << "\" (";
    for (auto [n, v]: d.values()) {
        o << "(\"" << n << "\" " << v << ")";
    }
    return o << "))";
}
std::ostream& sstring(std::ostream& o, const init_membrane_potential& p) {
    return o << "(membrane-potential " << p.value << ")";
}
std::ostream& sstring(std::ostream& o, const axial_resistivity& r) {
    return o << "(axial-resistivity " << r.value << ")";
}
std::ostream& sstring(std::ostream& o, const temperature_K& t) {
    return o << "(temperature-kelvin " << t.value << ")";
}
std::ostream& sstring(std::ostream& o, const membrane_capacitance& c) {
    return o << "(membrane-capacitance " << c.value << ")";
}
std::ostream& sstring(std::ostream& o, const init_int_concentration& c) {
    return o << "(ion-internal-concentration \"" << c.ion << "\" " << c.value << ")";
}
std::ostream& sstring(std::ostream& o, const init_ext_concentration& c) {
    return o << "(ion-external-concentration \"" << c.ion << "\" " << c.value << ")";
}
std::ostream& sstring(std::ostream& o, const init_reversal_potential& e) {
    return o << "(ion-reversal-potential \"" << e.ion << "\" " << e.value << ")";
}
std::ostream& sstring(std::ostream& o, const i_clamp& c) {
    return o << "(current-clamp " << c.amplitude << " " << c.delay << " " << c.duration << ")";
}
std::ostream& sstring(std::ostream& o, const threshold_detector& d) {
    return o << "(threshold-detector " << d.threshold << ")";
}
std::ostream& sstring(std::ostream& o, const gap_junction_site& s) {
    return o << "(gap-junciton-site)";
}

std::ostream& operator<<(std::ostream& o, const paintable& thing) {
    return std::visit([&o](auto&& p) -> std::ostream& {return sstring(o, p);}, thing);
}

std::ostream& operator<<(std::ostream& o, const placeable& thing) {
    return std::visit([&o](auto&& p) -> std::ostream& {return sstring(o, p);}, thing);
}

} // namespace arb
