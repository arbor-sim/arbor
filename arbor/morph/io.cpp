#include <functional>
#include <sstream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/io.hpp>

#include "s_expr.hpp"
#include "util/span.hpp"
#include "util/transform.hpp"

namespace arb {

// Helper functions that convert types into s-expressions.

// The paintables, placeables, defaultables...

using namespace s_expr_literals;

template <typename U, typename V>
s_expr mksexp(const std::pair<U, V>& p) {
    return slist(p.first, p.second);
}

struct as_s_expr {
    template <typename T>
    s_expr operator()(const T& v) const {
        return mksexp(v);
    }
};

s_expr mksexp(const init_membrane_potential& p) {
    return slist("membrane-potential"_symbol, p.value);
}

s_expr mksexp(const axial_resistivity& r) {
    return slist("axial-resistivity"_symbol, r.value);
}

s_expr mksexp(const temperature_K& t) {
    return slist("temperature-kelvin"_symbol, t.value);
}

s_expr mksexp(const membrane_capacitance& c) {
    return slist("membrane-capacitance"_symbol, c.value);
}

s_expr mksexp(const init_int_concentration& c) {
    return slist("ion-internal-concentration"_symbol, c.ion, c.value);
}

s_expr mksexp(const init_ext_concentration& c) {
    return slist("ion-external-concentration"_symbol, c.ion, c.value);
}

s_expr mksexp(const init_reversal_potential& e) {
    return slist("ion-reversal-potential"_symbol, e.ion, e.value);
}

s_expr mksexp(const mechanism_desc& d); // forward declaration
s_expr mksexp(const ion_reversal_potential_method& e) {
    return slist("ion-reversal-potential-method"_symbol, e.ion, mksexp(e.method));
}

s_expr mksexp(const i_clamp& c) {
    return slist("current-clamp"_symbol, c.amplitude, c.delay, c.duration);
}

s_expr mksexp(const threshold_detector& d) {
    return slist("threshold-detector"_symbol, d.threshold);
}

s_expr mksexp(const gap_junction_site& s) {
    return slist("gap-junction-site"_symbol);
}

s_expr mksexp(const initial_ion_data& s) {
    return slist("todo--ion-data"_symbol);
}

s_expr mksexp(const mechanism_desc& d) {
    using util::transform_view;
    return s_expr{"mechanism"_symbol, slist(d.name(), slist_range(transform_view(d.values(), as_s_expr())))};
}

// decorations on a cell
s_expr mksexp(const decor& d) {
    auto round_trip = [] (auto& x) {
        std::stringstream s;
        s << x;
        return parse_s_expr(s.str());
    };
    s_expr lst = slist();
    for (const auto& p: d.defaults) {
        lst = {std::visit([&](auto& x) {return slist("default"_symbol, mksexp(x));}, p), std::move(lst)};
    }
    for (const auto& p: d.paintings) {
        lst = {std::visit([&](auto& x) {return slist("paint"_symbol, round_trip(p.first), mksexp(x));}, p.second), std::move(lst)};
    }
    for (const auto& p: d.placements) {
        lst = {std::visit([&](auto& x) {return slist("place"_symbol, round_trip(p.first), mksexp(x));}, p.second), std::move(lst)};
    }
    return {"decorations"_symbol, std::move(lst)};
}

// label dictionary
s_expr mksexp(const label_dict& dict) {
    using namespace arb::s_expr_literals;
    auto round_trip = [] (auto& x) {
        std::stringstream s;
        s << x;
        return parse_s_expr(s.str());
    };

    auto defs = slist();
    for (auto& r: dict.regions()) {
        defs = s_expr(slist("region-def"_symbol, r.first, round_trip(r.second)), std::move(defs));
    }
    for (auto& r: dict.locsets()) {
        defs = s_expr(slist("locset-def"_symbol, r.first, round_trip(r.second)), std::move(defs));
    }

    return {"label-dict"_symbol, std::move(defs)};
}

s_expr mksexp(const mpoint& p) {
    return slist("point"_symbol, p.x, p.y, p.z, p.radius);
}

s_expr mksexp(const msegment& seg) {
    return slist("segment"_symbol, (int)seg.id, mksexp(seg.prox), mksexp(seg.dist), seg.tag);
}

s_expr mksexp(const morphology& morph) {
    // Range of morphology branches represented as s-expressions from an input morphology
    auto branches = [] (auto& m) {
        // s-expression representation of branch i in the morphology
        auto branch_description = [&m] (int i) {
            // List of msegments represented as s-expressions from an msegment sequence.
            auto seglist = [](auto& segs) {
                return slist_range(util::transform_view(segs, as_s_expr()));
            };
            return s_expr{"branch"_symbol, {i, {(int)m.branch_parent(i), seglist(m.branch_segments(i))}}};
        };
        auto index = util::make_span(m.num_branches());
        return util::transform_view(index, branch_description);
    };
    return s_expr{"morphology"_symbol, slist_range(branches(morph))};
}

std::ostream& write_s_expr(std::ostream& o, const label_dict& dict) {
    return o << mksexp(dict);
}

std::ostream& write_s_expr(std::ostream& o, const decor& decorations) {
    return o << mksexp(decorations);
}

std::ostream& write_s_expr(std::ostream& o, const cable_cell& c) {
    return o << s_expr{"cable-cell"_symbol, slist(mksexp(c.morphology()), mksexp(c.labels()), mksexp(c.decorations()))};
}

} // namespace arb
