#include <arbor/morph/label_parse.hpp>
#include <arbor/util/pp_util.hpp>
#include <arbor/util/any_visitor.hpp>

#include <arborio/cableio.hpp>

#include "parse_expression.hpp"

#include <iostream>

namespace arborio {

using namespace arb;

cableio_parse_error::cableio_parse_error(const std::string& msg, const arb::src_location& loc):
    arb::arbor_exception(msg+" at :"+std::to_string(loc.line)+":"+std::to_string(loc.column))
{}

struct nil_tag {};

// Define s-expr makers for various types
s_expr mksexp(const mpoint& p) {
    return slist("point"_symbol, p.x, p.y, p.z, p.radius);
}
s_expr mksexp(const msegment& seg) {
    return slist("segment"_symbol, (int)seg.id, mksexp(seg.prox), mksexp(seg.dist), seg.tag);
}
s_expr mksexp(const mechanism_desc& d) {
    std::vector<s_expr> mech;
    mech.push_back(d.name());
    for (const auto& p: d.values()) {
        mech.push_back(slist(p.first, p.second));
    }
    return s_expr{"mechanism"_symbol, slist_range(mech)};
}
s_expr mksexp(const ion_reversal_potential_method& e) {
    return slist("ion-reversal-potential-method"_symbol, e.ion, mksexp(e.method));
}
s_expr mksexp(const cv_policy& c) {
    return s_expr();
}
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
s_expr mksexp(const i_clamp& c) {
    return slist("current-clamp"_symbol, c.amplitude, c.delay, c.duration);
}
s_expr mksexp(const threshold_detector& d) {
    return slist("threshold-detector"_symbol, d.threshold);
}
s_expr mksexp(const gap_junction_site& s) {
    return slist("gap-junction-site"_symbol);
}
s_expr mksexp(const decor& d) {
    auto round_trip = [](auto& x) {
        std::stringstream s;
        s << x;
        return parse_s_expr(s.str());
    };
    std::vector<s_expr> decorations;
    for (const auto& p: d.defaults().serialize()) {
        decorations.push_back(std::visit([&](auto& x) { return slist("default"_symbol, mksexp(x)); }, p));
    }
    for (const auto& p: d.paintings()) {
        decorations.push_back(std::visit([&](auto& x) { return slist("paint"_symbol, round_trip(p.first), mksexp(x)); }, p.second));
    }
    for (const auto& p: d.placements()) {
        decorations.push_back(std::visit([&](auto& x) { return slist("place"_symbol, round_trip(p.first), mksexp(x)); }, p.second));
    }
    return {"decorations"_symbol, slist_range(decorations)};
}
s_expr mksexp(const label_dict& dict) {
    auto round_trip = [](auto& x) {
        std::stringstream s;
        s << x;
        return parse_s_expr(s.str());
    };
    auto defs = slist();
    for (auto& r: dict.locsets()) {
        defs = s_expr(slist("locset-def"_symbol, r.first, round_trip(r.second)), std::move(defs));
    }
    for (auto& r: dict.regions()) {
        defs = s_expr(slist("region-def"_symbol, r.first, round_trip(r.second)), std::move(defs));
    }
    return {"label-dict"_symbol, std::move(defs)};
}
s_expr mksexp(const morphology& morph) {
    // s-expression representation of branch i in the morphology
    auto make_branch = [&morph](int i) {
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
s_expr mksexp(const meta_data& meta) {
    return slist("meta-data"_symbol, slist("version"_symbol, meta.version));
}

// Implement public facing s-expr writers
std::ostream& write_component(std::ostream& o, const decor& x, const meta_data& m) {
    return o << s_expr{"arbor-component"_symbol, slist(mksexp(m), mksexp(x))};
}
std::ostream& write_component(std::ostream& o, const label_dict& x, const meta_data& m) {
    return o << s_expr{"arbor-component"_symbol, slist(mksexp(m), mksexp(x))};
}
std::ostream& write_component(std::ostream& o, const morphology& x, const meta_data& m) {
    return o << s_expr{"arbor-component"_symbol, slist(mksexp(m), mksexp(x))};
}
std::ostream& write_component(std::ostream& o, const cable_cell& x, const meta_data& m) {
    return o << s_expr{"arbor-component"_symbol, slist(mksexp(m), s_expr{"cable-cell"_symbol, slist(mksexp(x.morphology()), mksexp(x.labels()), mksexp(x.decorations()))})};
}
std::ostream& write_component(std::ostream& o, const cable_cell_component& x) {
    auto meta = x.meta;
    std::visit([&](auto&& c){write_component(o, c, meta);}, x.component);
    return o;
}

// Anonymous namespace containing helper functions and types for parsing s-expr
namespace {
// Test whether a value wrapped in std::any can be converted to a target type
template <typename T>
bool match(const std::type_info& info) {
    return info == typeid(T);
}
template <>
bool match<double>(const std::type_info& info) {
    return info == typeid(double) || info == typeid(int);
}

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

// Convert a value wrapped in a std::any to an optional std::variant type
template <typename T, std::size_t I=0>
std::optional<T> eval_cast_variant(const std::any& a) {
    if constexpr (I<std::variant_size_v<T>) {
        using var_type = std::variant_alternative_t<I, T>;
        return match<var_type>(a.type())? eval_cast<var_type>(a): eval_cast_variant<T, I+1>(a);
    }
    return std::nullopt;
}

// Define makers for defaultables, paintables, placeables
#define ARBIO_DEFINE_SINGLE_ARG(name) arb::name make_##name(double val) { return arb::name{val}; }
#define ARBIO_DEFINE_DOUBLE_ARG(name) arb::name make_##name(const std::string& ion, double val) { return arb::name{ion, val};}

ARB_PP_FOREACH(ARBIO_DEFINE_SINGLE_ARG, init_membrane_potential, temperature_K, axial_resistivity, membrane_capacitance, threshold_detector)
ARB_PP_FOREACH(ARBIO_DEFINE_DOUBLE_ARG, init_int_concentration, init_ext_concentration, init_reversal_potential)

arb::i_clamp make_i_clamp(double delay, double duration, double amplitude) {
    return arb::i_clamp(delay, duration, amplitude);
}
arb::gap_junction_site make_gap_junction_site() {
    return arb::gap_junction_site{};
}
arb::ion_reversal_potential_method make_ion_reversal_potential_method(const std::string& ion, const arb::mechanism_desc& mech) {
    return ion_reversal_potential_method{ion, mech};
}
#undef ARBIO_DEFINE_SINGLE_ARG
#undef ARBIO_DEFINE_DOUBLE_ARG

// Define makers for placeable pairs, paintable pairs, defaultables and decors
using place_pair = std::pair<arb::locset, arb::placeable>;
using paint_pair = std::pair<arb::region, arb::paintable>;
place_pair make_place(locset where, placeable what) {
    return place_pair{where, what};
}
paint_pair make_paint(region where, paintable what) {
    return paint_pair{where, what};
}
defaultable make_default(defaultable what) {
    return what;
}
decor make_decor(const std::vector<std::variant<place_pair, paint_pair, defaultable>>& args) {
    decor d;
    for(const auto& a: args) {
        auto decor_visitor = arb::util::overload(
            [&](const place_pair & p) { d.place(p.first, p.second); },
            [&](const paint_pair & p) { d.paint(p.first, p.second); },
            [&](const defaultable & p){ d.set_default(p); });
        std::visit(decor_visitor, a);
    }
    return d;
}

// Define maker for locset pairs, region pairs and label_dicts
using locset_pair = std::pair<std::string, locset>;
using region_pair = std::pair<std::string, region>;
locset_pair make_locset_pair(std::string name, locset desc) {
    return locset_pair{name, desc};
}
region_pair make_region_pair(std::string name, region desc) {
    return region_pair{name, desc};
}
label_dict make_label_dict(const std::vector<std::variant<locset_pair, region_pair>>& args) {
    label_dict d;
    for(const auto& a: args) {
        auto label_dict_visitor = arb::util::overload(
            [&](const locset_pair& p) { d.set(p.first, p.second); },
            [&](const region_pair& p) { d.set(p.first, p.second); });
        std::visit(label_dict_visitor, a);
    }
    return d;
}
// Define makers for mpoints and msegments and morphologies
using branch = std::tuple<int, int, std::vector<arb::msegment>>;
arb::mpoint make_point(double x, double y, double z, double r) {
    return arb::mpoint{x, y, z, r};
}
arb::msegment make_segment(unsigned id, arb::mpoint prox, arb::mpoint dist, int tag) {
    return arb::msegment{id, prox, dist, tag};
}
morphology make_morphology(const std::vector<std::variant<branch>>& args) {
    segment_tree tree;
    std::vector<unsigned> branch_final_seg(args.size());
    std::vector<std::pair<msegment, int>> segs;
    for (const auto& br: args) {
        auto b = std::get<branch>(br);
        auto b_id = std::get<0>(b);
        auto b_pid = std::get<1>(b);
        auto b_segments = std::get<2>(b);

        auto s_pid = b_pid==-1? arb::mnpos: branch_final_seg[b_pid];
        for (const auto& s: b_segments) {
            segs.emplace_back(s, s_pid);
            s_pid = s.id;
        }
        branch_final_seg[b_id] = s_pid;
    }
    std::sort(segs.begin(), segs.end(), [](const auto& lhs, const auto& rhs){return lhs.first.id < rhs.first.id;});
    for (const auto& spair: segs) {
        auto seg = spair.first;
        auto s_pid = spair.second;
        tree.append(s_pid, seg.prox, seg.dist, seg.tag);
    }
    return morphology(tree);
}

// Define cable-cell maker
// Accepts the morphology, decor and label_dict arguments in any order as a vector
cable_cell make_cablecell(const std::vector<std::variant<morphology, label_dict, decor>>& args) {
    decor dec;
    label_dict dict;
    morphology morpho;
    for(const auto& a: args) {
        auto cable_cell_visitor = arb::util::overload(
            [&](const morphology & p) { morpho = p; },
            [&](const label_dict & p) { dict = p; },
            [&](const decor & p){ dec = p; });
        std::visit(cable_cell_visitor, a);
    }
    return cable_cell(morpho, dict, dec);
}
using version = std::tuple<int>;
version make_version(int v) {
    return version{v};
}
meta_data make_meta_data(version v) {
    return meta_data{std::get<0>(v)};
}
template <typename T>
cable_cell_component make_component(const meta_data& m, const T& d) {
    return cable_cell_component{m, d};
}

// Evaluator: member of make_call, make_arg_vec_call, make_mech_call, make_branch_call, make_unordered_call
struct evaluator {
    using any_vec = std::vector<std::any>;
    using eval_fn = std::function<std::any(any_vec)>;
    using args_fn = std::function<bool(const any_vec&)>;

    eval_fn eval;
    args_fn match_args;
    const char* message;

    evaluator(eval_fn f, args_fn a, const char* m):
        eval(std::move(f)),
        match_args(std::move(a)),
        message(m)
    {}

    std::any operator()(any_vec args) {
        return eval(std::move(args));
    }
};

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
// Evaluate a call to a function where the arguments are provided as a std::vector<std::any>.
// The arguments are expanded and converted to the correct types, as specified by Args.
template <typename... Args>
struct call_eval {
    using ftype = std::function<std::any(Args...)>;
    ftype f;
    call_eval(ftype f): f(std::move(f)) {}

    template<std::size_t... I>
    std::any expand_args_then_eval(const std::vector<std::any>& args, std::index_sequence<I...>) {
        return f(eval_cast<Args>(std::move(args[I]))...);
    }

    std::any operator()(const std::vector<std::any>& args) {
        return expand_args_then_eval(std::move(args), std::make_index_sequence<sizeof...(Args)>());
    }
};
// Wrap call_match and call_eval in an evaluator
template <typename... Args>
struct make_call {
    evaluator state;

    template <typename F>
    make_call(F&& f, const char* msg="call"):
        state(call_eval<Args...>(std::forward<F>(f)), call_match<Args...>(), msg)
    {}
    operator evaluator() const {
        return state;
    }
};

// Test whether a list of arguments passed as a std::vector<std::any> can be converted
// to a std::vector<std::variant<Args...>>.
//
// For example, the following would return true:
//
//  call_match<int, string>(vector<any(4), any(12), any(string("hello"))>)
template <typename... Args>
struct arg_vec_match {
    template <typename T, typename Q, typename... Rest>
    bool match_args_impl(const std::any& arg) const {
        return match<T>(arg.type()) || match_args_impl<Q, Rest...>(arg);
    }

    template <typename T>
    bool match_args_impl(const std::any& arg) const {
        return match<T>(arg.type());
    }

    bool operator()(const std::vector<std::any>& args) const {
        for (const auto& a: args) {
            if (!match_args_impl<Args...>(a)) return false;
        }
        return true;
    }
};
// Evaluate a call to a function where the arguments are provided as a std::vector<std::any>.
// The arguments are converted to std::variant<Args...> and passed to the function as a std::vector.
template <typename... Args>
struct arg_vec_eval {
    using ftype = std::function<std::any(std::vector<std::variant<Args...>>)>;
    ftype f;
    arg_vec_eval(ftype f): f(std::move(f)) {}

    std::any operator()(const std::vector<std::any>& args) {
        std::vector<std::variant<Args...>> vars;
        for (const auto& a: args) {
            vars.push_back(eval_cast_variant<std::variant<Args...>>(a).value());
        }
        return f(vars);
    }
};
// Wrap arg_vec_match and arg_vec_eval in an evaluator
template <typename... Args>
struct make_arg_vec_call {
    evaluator state;

    template <typename F>
    make_arg_vec_call(F&& f, const char* msg="argument vector"):
        state(arg_vec_eval<Args...>(std::forward<F>(f)), arg_vec_match<Args...>(), msg)
    {}
    operator evaluator() const {
        return state;
    }
};

// Test whether a list of arguments passed as a std::vector<std::any> can be converted
// to a string followed by any number of std::pair<std::string, double>
using param_pair = std::pair<std::string, double>;
struct mech_match {
    bool operator()(const std::vector<std::any>& args) const {
        if (!match<std::string>(args.front().type())) return false;
        for (auto it = args.begin()+1; it != args.end(); ++it) {
            if(!match<param_pair>(it->type())) return false;
        }
        return true;
    }
};
// Create a mechanism_desc from a std::vector<std::any>.
struct mech_eval {
    arb::mechanism_desc operator()(const std::vector<std::any>& args) {
        auto name = eval_cast<std::string>(args.front());
        arb::mechanism_desc mech(name);
        for (auto it = args.begin()+1; it != args.end(); ++it) {
            auto p = eval_cast<param_pair>(*it);
            mech.set(p.first, p.second);
        }
        return mech;
    }
};
// Wrap mech_match and mech_eval in an evaluator
struct make_mech_call {
    evaluator state;
    make_mech_call(const char* msg="mechanism"):
        state(mech_eval(), mech_match(), msg)
    {}
    operator evaluator() const {
        return state;
    }
};

// Test whether a list of arguments passed as a std::vector<std::any> can be converted
// to 2 integers followed by at least 1 msegment
struct branch_match {
    bool operator()(const std::vector<std::any>& args) const {
        auto it = args.begin();
        if (!match<int>(it++->type())) return false;
        if (!match<int>(it++->type()))  return false;
        if (it == args.end()) return false;
        for (; it != args.end(); ++it) {
            if(!match<arb::msegment>(it->type())) return false;
        }
        return true;
    }
};
// Create a `branch` from a std::vector<std::any>.
struct branch_eval {
    branch operator()(const std::vector<std::any>& args) {
        std::vector<msegment> segs;
        auto it = args.begin();
        auto id = eval_cast<int>(*it++);
        auto parent = eval_cast<int>(*it++);
        for (; it != args.end(); ++it) {
            segs.push_back(eval_cast<msegment>(*it));
        }
        return branch{id, parent, segs};
    }
};
// Wrap branch_match and branch_eval in an evaluator
struct make_branch_call {
    evaluator state;
    make_branch_call(const char* msg="branch"):
        state(branch_eval(), branch_match(), msg)
    {}
    operator evaluator() const {
        return state;
    }
};

// Test whether a list of arguments passed as a std::vector<std::any> `args` can be
// converted to a std::vector<std::variant<Args...>>.
// - `args` must have the same size as Args...
// - no more than one element in `args` can match a given template parameter of Args...
//
// For example, the following would return true:
//
// call_match<int, string>(vector<any(4), any(string("hello"))>)
// call_match<int, string>(vector<any(string("hello")), any(4)>)
//
// And the following would return false:
//
// call_match<int, string>(vector<any(4), any(string("hello")), any(string("bye"))>)
// call_match<int, string>(vector<any(4), any(2)>)
//
// Not an efficient implementation, but should be okay for a few arguments.
template <typename... Args>
struct unordered_match {
    template <typename T, typename Q, typename... Rest>
    bool match_args_impl(const std::vector<std::any>& args) const {
        bool found_match = false;
        for (const auto& a: args) {
            auto new_match = match<T>(a.type());
            if (new_match && found_match) return false;
            found_match = new_match;
        }
        return found_match || match_args_impl<Q, Rest...>(args);
    }

    template <typename T>
    bool match_args_impl(const std::vector<std::any>& args) const {
        bool found_match = false;
        for (const auto& a: args) {
            auto new_match = match<T>(a.type());
            if (new_match && found_match) return false;
            found_match = new_match;
        }
        return found_match;
    }

    bool operator()(const std::vector<std::any>& args) const {
        const auto nargs_in = args.size();
        const auto nargs_ex = sizeof...(Args);
        return (nargs_in == nargs_ex) && match_args_impl<Args...>(args);
    }
};
// Wrap unordered_match and arg_vec_eval in an evaluator
template <typename... Args>
struct make_unordered_call {
    evaluator state;

    template <typename F>
    make_unordered_call(F&& f, const char* msg="call"):
        state(arg_vec_eval<Args...>(std::forward<F>(f)), unordered_match<Args...>(), msg)
    {}
    operator evaluator() const {
        return state;
    }
};
} // anonymous namespace

using eval_map = std::unordered_multimap<std::string, evaluator>;

// Parse s-expression into std::any given a function evaluation map.
parse_hopefully<std::any> eval(const arb::s_expr&, const eval_map&);

parse_hopefully<std::vector<std::any>> eval_args(const s_expr& e, const eval_map& map) {
    if (!e) return {std::vector<std::any>{}};
    std::vector<std::any> args;
    for (auto& h: e) {
        if (auto arg=eval(h, map)) {
            args.push_back(std::move(*arg));
        }
        else {
            return util::unexpected(std::move(arg.error()));
        }
    }
    return args;
}

parse_hopefully<std::any> eval(const s_expr& e, const eval_map& map ) {
    if (e.is_atom()) {
        auto& t = e.atom();
        switch (t.kind) {
        case tok::integer:
            return {std::stoi(t.spelling)};
        case tok::real:
            return {std::stod(t.spelling)};
        case tok::nil:
            return {nil_tag()};
        case tok::string:
            return std::any{std::string(t.spelling)};
            // An arbitrary symbol in a region/locset expression is an error, and is
            // often a result of not quoting a label correctly.
        case tok::symbol:
            return util::unexpected(cableio_parse_error("Unexpected symbol "+e.atom().spelling, location(e)));
        case tok::error:
            return util::unexpected(cableio_parse_error("Unexpected term "+e.atom().spelling, location(e)));
        default:
            return util::unexpected(cableio_parse_error("Unexpected term "+e.atom().spelling, location(e)));
        }
    }
    if (e.head().is_atom()) {
        // If this is a string, it must be a parameter pair of the form ("param" val)
        // where val is a double or int
        if (e.head().atom().kind == tok::string) {
            auto args = eval_args(e.tail(), map);
            if (!args) {
                return util::unexpected(args.error());
            }
            if (args->size() != 1) {
                return util::unexpected(cableio_parse_error("Expected parameter pair of the form (param:string val:real). "
                                                              "Got more than 1 `val` for `param` \"" + e.head().atom().spelling +"\".", location(e)));
            }
            if (!match<double>(args->front().type())) {
                return util::unexpected(cableio_parse_error("Expected parameter pair of the form (param:string val:real). "
                                                              "Got a `val` with a non-real type for `param` \"" + e.head().atom().spelling +"\".",location(e)));
            }
            return std::any{param_pair{e.head().atom().spelling, eval_cast<double>(args->front())}};
        };

        // Otherwise this must be a function evaluation, where head is the function name,
        // and tail is a list of arguments.
        // Evaluate the arguments, and return error state if an error occurred.
        auto args = eval_args(e.tail(), map);
        if (!args) {
            return util::unexpected(args.error());
        }

        // Find all candidate functions that match the name of the function.
        auto& name = e.head().atom().spelling;
        auto matches = map.equal_range(name);

        // Search for a candidate that matches the argument list.
        for (auto i=matches.first; i!=matches.second; ++i) {
            if (i->second.match_args(*args)) { // found a match: evaluate and return.
                return i->second.eval(*args);
            }
        }

        // If it's not in the provided map, maybe it's a label expression
        // the corresponding parser is provided by the arbor lib
        if (auto l = parse_label_expression(e)) {
            if (match<region>(l->type())) return eval_cast<region>(l.value());
            if (match<locset>(l->type())) return eval_cast<locset>(l.value());
        }

        // Unable to find a match: try to return a helpful error message.
        const auto nc = std::distance(matches.first, matches.second);
        std::string msg = "No matches for found for "+name+" with "+std::to_string(args->size())+" arguments.\n"
                          "There are "+std::to_string(nc)+" potential candiates"+(nc?":":".");
        int count = 0;
        for (auto i=matches.first; i!=matches.second; ++i) {
            msg += "\n  Candidate "+std::to_string(++count)+": "+i->second.message;
        }
        return util::unexpected(cableio_parse_error(msg, location(e)));
    }
    return util::unexpected(cableio_parse_error("Expression is neither integer, real expression of the form (op <args>) or (\"param\", val)", location(e)));
}

eval_map map{
    {"membrane-potential", make_call<double>(make_init_membrane_potential,
                               "'membrane-potential' with 1 argument (val:real)")},
    {"temperature-kelvin", make_call<double>(make_temperature_K,
                               "'temperature-kelvin' with 1 argument (val:real)")},
    {"axial-resistivity", make_call<double>(make_axial_resistivity,
                              "'axial-resistivity' with 1 argument (val:real)")},
    {"membrane-capacitance", make_call<double>(make_membrane_capacitance,
                                 "'membrane-capacitance' with 1 argument (val:real)")},
    {"ion-internal-concentration", make_call<std::string, double>(make_init_int_concentration,
                                       "'ion_internal_concentration' with 2 arguments (ion:string val:real)")},
    {"ion-external-concentration", make_call<std::string, double>(make_init_ext_concentration,
                                       "'ion_external_concentration' with 2 arguments (ion:string val:real)")},
    {"ion-reversal-potential", make_call<std::string, double>(make_init_reversal_potential,
                                   "'ion_reversal_potential' with 2 arguments (ion:string val:real)")},
    {"current-clamp", make_call<double, double, double>(make_i_clamp,
                          "'current-clamp' with 3 arguments (delay:real duration:real amplitude:real)")},
    {"threshold-detector", make_call<double>(make_threshold_detector,
                               "'threshold-detector' with 1 argument (threshold:real)")},
    {"gap-junction-site", make_call<>(make_gap_junction_site,
                              "'gap-junction-site' with 0 arguments")},
    {"ion-reversal-potential-method", make_call<std::string, arb::mechanism_desc>(make_ion_reversal_potential_method,
                                          "'ion-reversal-potential-method' with 2 ""arguments (ion:string mech:mechanism)")},
    {"mechanism", make_mech_call("'mechanism' with a name argument, and 0 or more parameter settings"
                                      "(name:string (param:string val:real))")},

    {"place", make_call<locset, gap_junction_site>(make_place, "'place' with 2 arguments (locset gap-junction-site)")},
    {"place", make_call<locset, i_clamp>(make_place, "'place' with 2 arguments (locset current-clamp)")},
    {"place", make_call<locset, threshold_detector>(make_place, "'place' with 2 arguments (locset threshold-detector)")},
    {"place", make_call<locset, mechanism_desc>(make_place, "'place' with 2 arguments (locset mechanism)")},

    {"paint", make_call<region, init_membrane_potential>(make_paint, "'paint' with 2 arguments (region membrane-potential)")},
    {"paint", make_call<region, temperature_K>(make_paint, "'paint' with 2 arguments (region temperature-kelvin)")},
    {"paint", make_call<region, membrane_capacitance>(make_paint, "'paint' with 2 arguments (region membrane-capacitance)")},
    {"paint", make_call<region, axial_resistivity>(make_paint, "'paint' with 2 arguments (region axial-resistivity)")},
    {"paint", make_call<region, init_int_concentration>(make_paint, "'paint' with 2 arguments (region ion-internal-concentration)")},
    {"paint", make_call<region, init_ext_concentration>(make_paint, "'paint' with 2 arguments (region ion-external-concentration)")},
    {"paint", make_call<region, init_reversal_potential>(make_paint, "'paint' with 2 arguments (region ion-reversal-potential)")},
    {"paint", make_call<region, mechanism_desc>(make_paint, "'paint' with 2 arguments (region mechanism)")},

    {"default", make_call<init_membrane_potential>(make_default, "'default' with 1 argument (membrane-potential)")},
    {"default", make_call<temperature_K>(make_default, "'default' with 1 argument (temperature-kelvin)")},
    {"default", make_call<membrane_capacitance>(make_default, "'default' with 1 argument (membrane-capacitance)")},
    {"default", make_call<axial_resistivity>(make_default, "'default' with 1 argument (axial-resistivity)")},
    {"default", make_call<init_int_concentration>(make_default, "'default' with 1 argument (ion-internal-concentration)")},
    {"default", make_call<init_ext_concentration>(make_default, "'default' with 1 argument (ion-external-concentration)")},
    {"default", make_call<init_reversal_potential>(make_default, "'default' with 1 argument (ion-reversal-potential)")},
    {"default", make_call<ion_reversal_potential_method>(make_default, "'default' with 1 argument (ion-reversal-potential-method)")},

    {"locset-def", make_call<std::string, locset>(make_locset_pair,
                       "'locset-def' with 2 arguments (name:string ls:locset)")},
    {"region-def", make_call<std::string, region>(make_region_pair,
                       "'region-def' with 2 arguments (name:string reg:region)")},

    {"point",   make_call<double, double, double, double>(make_point,
                    "'point' with 4 arguments (x:real y:real z:real radius:real)")},
    {"segment", make_call<int, mpoint, mpoint, int>(make_segment,
                    "'segment' with 4 arguments (parent:int prox:point dist:point tag:int)")},
    {"branch",  make_branch_call(
                    "'branch' with 2 integers and 1 or more segment arguments (id:int parent:int s0:segment s1:segment ..)")},

    {"decorations", make_arg_vec_call<place_pair, paint_pair, defaultable>(make_decor,
                        "'decorations' with 1 or more `paint`, `place` or `default` arguments")},
    {"label-dict", make_arg_vec_call<locset_pair, region_pair>(make_label_dict,
                       "'label-dict' with 1 or more `locset-def` or `region-def` arguments")},
    {"morphology", make_arg_vec_call<branch>(make_morphology,
                       "'morphology' 1 or more `branch` arguments")},

    {"cable-cell", make_unordered_call<morphology, label_dict, decor>(make_cablecell,
                       "'cable-cell' with 3 arguments: `morphology`, `label-dict`, and `decor` in any order")},

    {"version", make_call<int>(make_version, "'version' with one argment (val:int)")},
    {"meta-data", make_call<version>(make_meta_data, "'meta-data' with one argument (v:version)")},

    { "arbor-component", make_call<meta_data, decor>(make_component<decor>, "'arbor-component' with 2 arguments (m:meta_data p:decoration)")},
    { "arbor-component", make_call<meta_data, label_dict>(make_component<label_dict>, "'arbor-component' with 2 arguments (m:meta_data p:decoration)")},
    { "arbor-component", make_call<meta_data, morphology>(make_component<morphology>, "'arbor-component' with 2 arguments (m:meta_data p:label_dict)")},
    { "arbor-component", make_call<meta_data, cable_cell>(make_component<cable_cell>, "'arbor-component' with 2 arguments (m:meta_data p:cable_cell)")}
};

inline parse_hopefully<std::any> parse(const arb::s_expr& s) {
    return eval(std::move(s), map);
}

parse_hopefully<std::any> parse_expression(const std::string& s) {
    return parse(parse_s_expr(s));
}

// Read s-expr
parse_hopefully<cable_cell_component> parse_component(const std::string& s) {
    auto try_parse = parse(parse_s_expr(s));
    if (!try_parse) {
        return util::unexpected(cableio_parse_error(try_parse.error()));
    }
    if (!match<cable_cell_component>(try_parse.value().type())) {
        return util::unexpected(cableio_parse_error("Expected arbor-component", location(s)));
    }
    auto comp = eval_cast<cable_cell_component>(try_parse.value());
    if (comp.meta.version != CABLE_CELL_FORMAT_VERSION) {
        return util::unexpected(cableio_parse_error("Unsupported cable-cell format version "+std::to_string(comp.meta.version), location(s)));
    }
    return comp;
};

parse_hopefully<cable_cell_component> parse_component(std::istream& s) {
    return parse_component(std::string(std::istreambuf_iterator<char>(s), {}));
}
} // namespace arborio