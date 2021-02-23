#include <iostream>
#include <arbor/util/pp_util.hpp>
#include <arbor/util/any_visitor.hpp>

#include "cable_cell_format.hpp"

namespace arborio {

using namespace arb;

// S-expression makers
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
    s_expr lst = slist();
    for (const auto& p: d.defaults().serialize()) {
        lst = {std::visit([&](auto& x) { return slist("default"_symbol, mksexp(x)); }, p), std::move(lst)};
    }
    for (const auto& p: d.paintings()) {
        lst = {std::visit([&](auto& x) { return slist("paint"_symbol, round_trip(p.first), mksexp(x)); }, p.second), std::move(lst)};
    }
    for (const auto& p: d.placements()) {
        lst = {std::visit([&](auto& x) { return slist("place"_symbol, round_trip(p.first), mksexp(x)); }, p.second), std::move(lst)};
    }
    return {"decorations"_symbol, std::move(lst)};
}
s_expr mksexp(const label_dict& dict) {
    auto round_trip = [](auto& x) {
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

// Anonymous namespace containing helper functions and types
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

template <typename T, std::size_t I=0>
std::optional<T> eval_cast_variant(const std::any& a) {
    if constexpr (I<std::variant_size_v<T>) {
        using var_type = std::variant_alternative_t<I, T>;
        return typeid(var_type) == a.type()? std::any_cast<var_type>(a): eval_cast_variant<T, I+1>(a);
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

// Define makers for place_pair, paint_pair and decor
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
decor make_decor(std::vector<std::variant<place_pair, paint_pair, defaultable>> args) {
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

// Define maker for locset_pair, region_pair and label_dict
using locset_pair = std::pair<std::string, locset>;
using region_pair = std::pair<std::string, region>;
locset_pair make_locset_pair(std::string name, locset desc) {
    return locset_pair{name, desc};
}
region_pair make_region_pair(std::string name, region desc) {
    return region_pair{name, desc};
}
label_dict make_label_dict(std::vector<std::variant<locset_pair, region_pair>> args) {
    label_dict d;
    for(const auto& a: args) {
        auto label_dict_visitor = arb::util::overload(
            [&](const locset_pair& p) { d.set(p.first, p.second); },
            [&](const region_pair& p) { d.set(p.first, p.second); });
        std::visit(label_dict_visitor, a);
    }
    return d;
}
// Define makers for mpoints and msegments and morphology
arb::mpoint make_point(double x, double y, double z, double r) {
    return arb::mpoint{x, y, z, r};
}
arb::msegment make_segment(unsigned id, arb::mpoint prox, arb::mpoint dist, int tag) {
    return arb::msegment{id, prox, dist, tag};
}
struct branch {
    int id;
    int parent_id;
    std::vector<arb::msegment> segments;
};

morphology make_morphology(std::vector<std::variant<branch>> args) {
    segment_tree tree;
    std::vector<unsigned> branch_final_seg(args.size());
    for (const auto& br: args) {
        auto b = std::get<branch>(br);
        auto pseg_id = b.parent_id==-1? arb::mnpos: branch_final_seg[b.parent_id];
        for (const auto& s: b.segments) {
            pseg_id = tree.append(pseg_id, s.prox, s.dist, s.tag);
        }
        branch_final_seg[b.id] = pseg_id;
    }
    return morphology(tree);
}

// Define cable-cell maker
cable_cell make_cable(morphology morpho, label_dict dict, decor dec) {
   return cable_cell(morpho, dict, dec);
}

// Evaluation
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
    std::any expand_args_then_eval(std::vector<std::any> args, std::index_sequence<I...>) {
        return f(eval_cast<Args>(std::move(args[I]))...);
    }

    std::any operator()(std::vector<std::any> args) {
        return expand_args_then_eval(std::move(args), std::make_index_sequence<sizeof...(Args)>());
    }
};
// wrap call_match and call_eval
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
template <typename... Args>
struct arg_vec_eval {
    using ftype = std::function<std::any(std::vector<std::variant<Args...>>)>;
    ftype f;
    arg_vec_eval(ftype f): f(std::move(f)) {}

    std::any operator()(std::vector<std::any> args) {
        std::vector<std::variant<Args...>> vars;
        for (const auto& a: args) {
            vars.push_back(eval_cast_variant<std::variant<Args...>>(a).value());
        }
        return f(vars);
    }
};
template <typename... Args>
struct make_arg_vec_call {
    evaluator state;

    template <typename F>
    make_arg_vec_call(F&& f, const char* msg="arg_vec"):
        state(arg_vec_eval<Args...>(std::forward<F>(f)), arg_vec_match<Args...>(), msg)
    {}
    operator evaluator() const {
        return state;
    }
};

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
struct mech_eval {
    arb::mechanism_desc operator()(std::vector<std::any> args) {
        auto name = eval_cast<std::string>(args.front());
        arb::mechanism_desc mech(name);
        for (auto it = args.begin()+1; it != args.end(); ++it) {
            auto p = eval_cast<param_pair>(*it);
            mech.set(p.first, p.second);
        }
        return mech;
    }
};
struct make_mech_call {
    evaluator state;
    make_mech_call(const char* msg="arg_vec"):
        state(mech_eval(), mech_match(), msg)
    {}
    operator evaluator() const {
        return state;
    }
};

struct branch_match {
    bool operator()(const std::vector<std::any>& args) const {
        auto it = args.begin();
        if (!match<int>(it++->type())) return false;
        if (!match<int>(it++->type()))  return false;
        for (; it != args.end(); ++it) {
            if(!match<arb::msegment>(it->type())) return false;
        }
        return true;
    }
};
struct branch_eval {
    branch operator()(std::vector<std::any> args) {
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
struct make_branch_call {
    evaluator state;
    make_branch_call(const char* msg="arg_vec"):
        state(branch_eval(), branch_match(), msg)
    {}
    operator evaluator() const {
        return state;
    }
};
} // anonymous namespace

// Parse s-expression into std::any based on function eval_map
using eval_map = std::unordered_multimap<std::string, evaluator>;
parse_hopefully<std::any> eval(const arb::s_expr&, const eval_map&);

parse_hopefully<std::vector<std::any>> eval_args(const s_expr& e, const eval_map& map) {
    if (!e) return {std::vector<std::any>{}}; // empty argument list
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
            return util::unexpected(cableio_unexpected_symbol(e.atom().spelling, location(e)));
        case tok::error:
            return util::unexpected(cableio_parse_error(e.atom().spelling, location(e)));
        default:
            return util::unexpected(cableio_parse_error("Unexpected term "+e.atom().spelling, location(e)));
        }
    }
    if (e.head().is_atom()) {
        // If this is a string, it must be a parameter pair
        if (e.head().atom().kind == tok::string) {
            auto args = eval_args(e.tail(), map);
            if (!args) {
                return util::unexpected(args.error());
            }
            if (args->size() != 1 || (!match<int>(args->front().type()) && !match<double>(args->front().type()))) {
                return util::unexpected(cableio_parse_error("Parameter "+e.head().atom().spelling+" can only have an integer or real value.", location(e)));
            }
            return std::any{param_pair{e.head().atom().spelling, eval_cast<double>(args->front())}};
        };
        // This must be a function evaluation, where head is the function name, and
        // tail is a list of arguments.

        // Evaluate the arguments, and return error state if an error ocurred.
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
        std::cout << e << std::endl;
        return util::unexpected(cableio_parse_error("No matches for "+name, location(e)));
    }
    return util::unexpected(cableio_parse_error("expression is neither integer, real expression of the form (op <args>)", location(e)));
}

parse_hopefully<std::any> parse_expression(const arb::s_expr& s) {
    eval_map map{
        {"membrane-potential", make_call<double>(make_init_membrane_potential, "'membrane-potential' with 1 argument")},
        {"temperature-kelvin", make_call<double>(make_temperature_K, "'temperature-kelvin' with 1 argument")},
        {"axial-resistivity", make_call<double>(make_axial_resistivity, "'axial-resistivity' with 1 argument")},
        {"membrane-capacitance", make_call<double>(make_membrane_capacitance, "'membrane-capacitance' with 1 argument")},
        {"ion-internal-concentration", make_call<std::string, double>(make_init_int_concentration, "'ion_internal_concentration' with 2 arguments")},
        {"ion-external-concentration", make_call<std::string, double>(make_init_ext_concentration, "'ion_external_concentration' with 2 arguments")},
        {"ion-reversal-potential", make_call<std::string, double>(make_init_reversal_potential, "'ion_reversal_potential' with 2 arguments")},
        {"current-clamp", make_call<double, double, double>(make_i_clamp, "'current-clamp' with 3 arguments")},
        {"threshold-detector", make_call<double>(make_threshold_detector, "'threshold-detector' with 1 argument")},
        {"gap-junction-site", make_call<>(make_gap_junction_site, "'gap-junction-site' with 0 arguments")},
        {"ion-reversal-potential-method", make_call<std::string, arb::mechanism_desc>(make_ion_reversal_potential_method, "'ion-reversal-potential-method' with 2 arguments")},
        {"mechanism", make_mech_call("'mechanism' with at least one argument")},

        {"place", make_call<locset, gap_junction_site>(make_place, "'place' with two arguments")},
        {"place", make_call<locset, i_clamp>(make_place, "'place' with two arguments")},
        {"place", make_call<locset, threshold_detector>(make_place, "'place' with two arguments")},
        {"place", make_call<locset, mechanism_desc>(make_place, "'place' with two arguments")},

        {"paint", make_call<region, init_membrane_potential>(make_paint, "'paint' with two arguments")},
        {"paint", make_call<region, temperature_K>(make_paint, "'paint' with two arguments")},
        {"paint", make_call<region, membrane_capacitance>(make_paint, "'paint' with two arguments")},
        {"paint", make_call<region, axial_resistivity>(make_paint, "'paint' with two arguments")},
        {"paint", make_call<region, init_int_concentration>(make_paint, "'paint' with two arguments")},
        {"paint", make_call<region, init_ext_concentration>(make_paint, "'paint' with two arguments")},
        {"paint", make_call<region, init_reversal_potential>(make_paint, "'paint' with two arguments")},
        {"paint", make_call<region, mechanism_desc>(make_paint, "'paint' with two arguments")},

        {"default", make_call<init_membrane_potential>(make_default, "'default' with one argument")},
        {"default", make_call<temperature_K>(make_default, "'default' with one argument")},
        {"default", make_call<membrane_capacitance>(make_default, "'default' with one argument")},
        {"default", make_call<axial_resistivity>(make_default, "'default' with one argument")},
        {"default", make_call<init_int_concentration>(make_default, "'default' with one argument")},
        {"default", make_call<init_ext_concentration>(make_default, "'default' with one argument")},
        {"default", make_call<init_reversal_potential>(make_default, "'default' with one argument")},
        {"default", make_call<ion_reversal_potential_method>(make_default, "'default' with one argument")},

        {"locset-def", make_call<std::string, locset>(make_locset_pair, "'locset-def' with 2 arguments")},
        {"region-def", make_call<std::string, region>(make_region_pair, "'region-def' with 2 arguments")},

        {"point",   make_call<double, double, double, double>(make_point, "'point' with 4 arguments")},
        {"segment", make_call<int, mpoint, mpoint, int>(make_segment, "'segment' with 4 arguments")},
        {"branch",  make_branch_call("'branch' with at least 1 argument")},

        {"decorations", make_arg_vec_call<place_pair, paint_pair, defaultable>(make_decor,"'decor' with at least one argument")},
        {"label-dict", make_arg_vec_call<locset_pair, region_pair>(make_label_dict, "'label-dict' with at least 1 argument")},
        {"morphology", make_arg_vec_call<branch>(make_morphology,"'morphology' with at least 1 argument")},

        {"cable-cell", make_call<morphology, label_dict, decor>(make_cable, "'cable-cell' with at least 1 argument")}
    };
    return eval(std::move(s), map);
}

using cable_cell_components = std::variant<morphology, label_dict, decor, cable_cell>;
std::optional<cable_cell_components> parse(const arb::s_expr& s) {
    return eval_cast_variant<cable_cell_components>(parse_expression(s));
};

} // namespace arborio