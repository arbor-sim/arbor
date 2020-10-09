#include <iostream>

#include <cfloat>
#include <cmath>
#include <numeric>
#include <vector>
#include <variant>

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>

#include "util/maputil.hpp"
#include "s_expr.hpp"

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


// s-expression printers for paintable, placeable and defaultable items.

s_expr make_s_expr(const init_membrane_potential& p) {
    using namespace s_expr_literals;
    return slist("membrane-potential"_symbol, p.value);
}
s_expr make_s_expr(const axial_resistivity& r) {
    using namespace s_expr_literals;
    return slist("axial-resistivity"_symbol, r.value);
}
s_expr make_s_expr(const temperature_K& t) {
    using namespace s_expr_literals;
    return slist("temperature-kelvin"_symbol, t.value);
}
s_expr make_s_expr(const membrane_capacitance& c) {
    using namespace s_expr_literals;
    return slist("membrane-capacitance"_symbol, c.value);
}
s_expr make_s_expr(const init_int_concentration& c) {
    using namespace s_expr_literals;
    return slist("ion-internal-concentration"_symbol, c.ion, c.value);
}
s_expr make_s_expr(const init_ext_concentration& c) {
    using namespace s_expr_literals;
    return slist("ion-external-concentration"_symbol, c.ion, c.value);
}
s_expr make_s_expr(const init_reversal_potential& e) {
    using namespace s_expr_literals;
    return slist("ion-reversal-potential"_symbol, e.ion, e.value);
}
s_expr make_s_expr(const mechanism_desc& d); // forward declaration
s_expr make_s_expr(const ion_reversal_potential_method& e) {
    using namespace s_expr_literals;
    return slist("ion-reversal-potential-method"_symbol, e.ion, make_s_expr(e.method));
}
s_expr make_s_expr(const i_clamp& c) {
    using namespace s_expr_literals;
    return slist("current-clamp"_symbol, c.amplitude, c.delay, c.duration);
}
s_expr make_s_expr(const threshold_detector& d) {
    using namespace s_expr_literals;
    return slist("threshold-detector"_symbol, d.threshold);
}
s_expr make_s_expr(const gap_junction_site& s) {
    using namespace s_expr_literals;
    return slist("gap-junction-site"_symbol);
}
s_expr make_s_expr(const initial_ion_data& s) {
    using namespace s_expr_literals;
    return slist("ion-data"_symbol);
}
s_expr make_s_expr(const mechanism_desc& d) {
    s_expr e;
    s_expr args = slist(); // initialise arguments as an empty list
    for (auto [n, v]: d.values()) {
        args = s_expr({n, v}, std::move(args));
    }
    return slist("mechanism", args);
}

std::ostream& operator<<(std::ostream& o, const decor& d) {
    o << "(defaults";
    for (const auto& p: d.defaults) {
        o << "\n  " << " " << std::visit([](auto& x) {return make_s_expr(x);}, p) << ")";
    }
    o << ")\n";
    o << "(paintings";
    for (const auto& p: d.paintings) {
        o << "\n  (" << p.first << " " << std::visit([](auto& x) {return make_s_expr(x);}, p.second) << ")";
    }
    o << ")\n";
    o << "(placements";
    for (const auto& p: d.placements) {
        o << "\n  (" << p.first << " " << std::visit([](auto& x) {return make_s_expr(x);}, p.second) << ")";
    }
    o << ")";
    return o;
}

} // namespace arb
