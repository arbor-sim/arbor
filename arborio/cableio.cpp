#include <iostream>
#include <numeric>
#include <functional>
#include <sstream>
#include <numeric>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/label_parse.hpp>
#include <arbor/s_expr.hpp>
#include <arbor/util/any_visitor.hpp>

#include <arborio/cableio.hpp>

namespace arborio{
using namespace arb;

// Errors
format_parse_error::format_parse_error(const std::string& msg):
    arb::arbor_exception(msg)
{}

// Helpers
inline symbol operator "" _symbol(const char* chars, size_t size) {
    return {chars};
}

// Write s-expr

// Helper functions that convert types into s-expressions.
template <typename U, typename V>
s_expr mksexp(const std::pair<U, V>& p) {
    return slist(p.first, p.second);
}

// Forward declarations
s_expr mksexp(const mechanism_desc&);
s_expr mksexp(const msegment&);

struct as_s_expr {
    template <typename T>
    s_expr operator()(const T& v) const {
        return mksexp(v);
    }
};

// Defaultable and paintable
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
s_expr mksexp(const initial_ion_data& s) {
    std::vector<s_expr> ion_data;
    if (auto iconc = s.initial.init_int_concentration) {
        ion_data.push_back(slist("ion-internal-concentration"_symbol, s.ion, iconc.value()));
    }
    if (auto econc = s.initial.init_ext_concentration) {
        ion_data.push_back(slist("ion-external-concentration"_symbol, s.ion, econc.value()));
    }
    if (auto revpot = s.initial.init_reversal_potential) {
        ion_data.push_back(slist("ion-reversal-potential"_symbol, s.ion, revpot.value()));
    }
    return slist_range(ion_data);
}

// Defaultable
s_expr mksexp(const ion_reversal_potential_method& e) {
    return slist("ion-reversal-potential-method"_symbol, e.ion, mksexp(e.method));
}
s_expr mksexp(const cv_policy& c) {
    return s_expr();
}

// Paintable
s_expr mksexp(const mechanism_desc& d) {
    std::vector<s_expr> params;
    for (const auto& p: d.values()) {
        params.push_back(mksexp(p));
    }
    return s_expr{"mechanism"_symbol, slist(d.name(), slist_range(params))};
}

// Placeable
s_expr mksexp(const i_clamp& c) {
    return slist("current-clamp"_symbol, c.amplitude, c.delay, c.duration);
}
s_expr mksexp(const threshold_detector& d) {
    return slist("threshold-detector"_symbol, d.threshold);
}
s_expr mksexp(const gap_junction_site& s) {
    return slist("gap-junction-site"_symbol);
}

// Decor
s_expr mksexp(const decor& d) {
    auto round_trip = [] (auto& x) {
      std::stringstream s;
      s << x;
      return parse_s_expr(s.str());
    };
    s_expr lst = slist();
    for (const auto& p: d.defaults().serialize()) {
        lst = {std::visit([&](auto& x) {return slist("default"_symbol, mksexp(x));}, p), std::move(lst)};
    }
    for (const auto& p: d.paintings()) {
        lst = {std::visit([&](auto& x) {return slist("paint"_symbol, round_trip(p.first), mksexp(x));}, p.second), std::move(lst)};
    }
    for (const auto& p: d.placements()) {
        lst = {std::visit([&](auto& x) {return slist("place"_symbol, round_trip(p.first), mksexp(x));}, p.second), std::move(lst)};
    }
    return {"decorations"_symbol, std::move(lst)};
}

// Label dictionary
s_expr mksexp(const label_dict& dict) {
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

// Morphology
s_expr mksexp(const mpoint& p) {
    return slist("point"_symbol, p.x, p.y, p.z, p.radius);
}
s_expr mksexp(const msegment& seg) {
    return slist("segment"_symbol, (int)seg.id, mksexp(seg.prox), mksexp(seg.dist), seg.tag);
}
s_expr mksexp(const morphology& morph) {
    // s-expression representation of branch i in the morphology
    auto make_branch = [&morph] (int i) {
        std::vector<s_expr> segments;
        for (const auto& s: morph.branch_segments(i)) {
            segments.push_back(mksexp(s));
        }
        return s_expr{"branch"_symbol, {i, {(int)morph.branch_parent(i), slist_range(segments)}}};
    };
    std::vector<s_expr> branches;
    for (msize_t i = 0; i < morph.num_branches(); ++i) {
      branches.push_back(make_branch(i));
    }
    return s_expr{"morphology"_symbol, slist_range(branches)};
}

std::ostream& write_s_expr(std::ostream& o, const label_dict& dict) {
    return o << mksexp(dict);
}
std::ostream& write_s_expr(std::ostream& o, const decor& decorations) {
    return o << mksexp(decorations);
}
std::ostream& write_s_expr(std::ostream& o, const morphology& morphology) {
    return o << mksexp(morphology);
}
std::ostream& write_s_expr(std::ostream& o, const cable_cell& c) {
    return o << s_expr{"cable-cell"_symbol, slist(mksexp(c.morphology()), mksexp(c.labels()), mksexp(c.decorations()))};
}

// Read s-expr

// Anonymous namespace containing helper functions and types
/*namespace {

// Test whether a value wrapped in std::any can be converted to a target type
template <typename T>
bool match(const std::type_info& info) {
    return info == typeid(T);
}
template <>
bool match<double>(const std::type_info& info) {
    return info == typeid(double) || info == typeid(int);
}

// Test whether a list of arguments passed as a std::vector<std::any> can be converted
// to the types in Args.
//
// For example, the following would return true:
//
//  call_match<int, int, string>(vector<any(4), any(12), any(string("hello"))>)
template <typename... Args>
struct call_match {
    template <std::size_t I, typename T, typename Q, typename... Rest>
    bool match_args_impl(const std::vector<std::any>& args) const {
        return match<T>(args[I].type()) && match_args_impl<I+1, Q, Rest...>(args);
    }

    template <std::size_t I, typename T>
    bool match_args_impl(const std::vector<std::any>& args) const {
        return match<T>(args[I].type());
    }

    template <std::size_t I>
    bool match_args_impl(const std::vector<std::any>& args) const {
        return true;
    }

    bool operator()(const std::vector<std::any>& args) const {
        const auto nargs_in = args.size();
        const auto nargs_ex = sizeof...(Args);
        return nargs_in==nargs_ex? match_args_impl<0, Args...>(args): false;
    }
};

// Convert a value wrapped in a std::any to target type.
template <typename T>
T eval_cast(std::any arg) {
    return std::move(std::any_cast<T&>(arg));
}
template <>
double eval_cast<double>(std::any arg) {
    if (arg.type()==typeid(int)) return std::any_cast<int>(arg);
    return std::any_cast<double>(arg);
}

// Evaluate a call to a function where the arguments are provided as a std::vector<std::any>.
// The arguments are expanded and converted to the correct types, as specified by Args.
template <typename... Args>
struct call_eval {
    using ftype = std::function<std::any(Args...)>;
    ftype f;
    call_eval(ftype f): f(std::move(f)) {}

    template<std::size_t... I>
    std::any expand_args_then_eval(std::vector<std::any> args, std::index_sequence<I...>) {
        return f(eval_cast<Args>(std::move(args[I]))...);
    }

    std::any operator()(std::vector<std::any> args) {
        return expand_args_then_eval(std::move(args), std::make_index_sequence<sizeof...(Args)>());
    }
};

struct evaluator {
    using any_vec = std::vector<std::any>;
    using eval_fn = std::function<std::any(any_vec)>;
    using args_fn = std::function<bool(const any_vec&)>;

    eval_fn function;
    args_fn match_args;
    const char* message;

    evaluator(eval_fn f, args_fn a, const char* m):
        function(std::move(f)),
        match_args(std::move(a)),
        message(m)
    {}

    std::any operator()(any_vec args) {
        return function(std::move(args));
    }
};

template <typename... Args>
struct define_call {
    evaluator state;

    template <typename F>
    define_call(F&& f, const char* msg="call"):
        state(call_eval<Args...>(std::forward<F>(f)), call_match<Args...>(), msg)
    {}

    operator evaluator() const {
        return state;
    }
};

}*/ // anonymous namespace

struct label_pair {
    std::string label;
    std::variant<region, locset> desc;
};

parse_hopefully<label_pair> eval_dict(const s_expr& e) {
    if (e.is_atom()) {
        return util::unexpected(format_parse_error("eval_dict expected atom"));
    }
    if (e.head().is_atom()) {
        auto& name = e.head().atom().spelling;
        if (name != "region-def" && name != "locset-def") {
            return util::unexpected(format_parse_error("eval_dict expected region-def or locset-def"));
        }
        auto args = e.tail();
        if (!args.head().is_atom() || args.head().atom().kind != tok::string) {
            return util::unexpected(format_parse_error("eval_dict arg expected string atom"));
        }
        if (args.tail().is_atom()) {
            return util::unexpected(format_parse_error("eval_dict arg did not expect string atom"));
        }
        if (!args.tail().tail().is_atom() || args.tail().tail().atom().kind != tok::nil) {
            return util::unexpected(format_parse_error("eval_dict expected bahh"));
        }
        label_pair p;
        p.label = args.head().atom().spelling;
        if (auto desc = parse_label_expression(args.tail().head())) {
            if (desc->type() == typeid(locset) && name == "locset-def") {
                p.desc = std::any_cast<locset&>(*desc);
                return p;
            }
            if (desc->type() == typeid(region) && name == "region-def") {
                p.desc = std::any_cast<region&>(*desc);
                return p;
            }
        } else {
            throw desc.error();
        }
    }
    return util::unexpected(format_parse_error("expected something else"));
}

parse_hopefully<label_dict> parse_label_dict(const std::string& str) {
    auto s = parse_s_expr(str);
    std::cout << length(s) << std::endl;
    if (!s.head().is_atom()) {
        throw format_parse_error("Expected atom at head");
    }
    if (s.head().atom().kind != tok::symbol) {
        throw format_parse_error("Expected symbol at head");
    }
    if (s.head().atom().spelling != "label-dict") {
        throw format_parse_error("Expected label-dict symbol at head");
    }
    label_dict d;
    for (const auto& t: s.tail()) {
        if (auto e = eval_dict(t)) {
            if (auto eval = std::get_if<region>(&e->desc)) d.set(e->label, *eval);
            if (auto eval = std::get_if<locset>(&e->desc)) d.set(e->label, *eval);
        }
        else {
            throw e.error();
        }
    }
    return d;
}
/*
{    label_dict d;
    for (auto& entry: e) {
        auto it = std::begin(entry);
        std::string kind = get<symbol>(*it);
        ++it;
        if (kind=="locset-def") {
            std::string name = get<std::string>(*it);
            ++it;
            region reg = parse_region_expression(*it);
            d.set(name, std::move(reg));
        }
        else {
            std::string name = get<std::string>(*it);
            ++it;
            region reg = parse_locset_expression(*it);
            d.set(name, std::move(reg));
        }
    }

    return d;
}
*/
} // namespace arborio
