#include <algorithm>
#include <sstream>
#include <unordered_map>

#include <arbor/s_expr.hpp>
#include <arbor/util/pp_util.hpp>
#include <arbor/util/any_visitor.hpp>
#include <arbor/iexpr.hpp>
#include <arbor/cable_cell_param.hpp>

#include <arborio/label_parse.hpp>
#include <arborio/cableio.hpp>
#include <arborio/cv_policy_parse.hpp>

#include "parse_helpers.hpp"
#include "parse_s_expr.hpp"

namespace arborio {

using namespace arb;

ARB_ARBORIO_API std::string acc_version() {return "0.1-dev";}

cableio_parse_error::cableio_parse_error(const std::string& msg, const arb::src_location& loc):
    arb::arbor_exception(msg+" at :"+
                         std::to_string(loc.line)+ ":"+
                         std::to_string(loc.column))
{}

cableio_morphology_error::cableio_morphology_error(const unsigned bid):
    arb::arbor_exception("Invalid morphology: branch `" +std::to_string(bid) +
                         "` only has one child branch, making it an invalid branch specification")
{}

cableio_version_error::cableio_version_error(const std::string& version):
    arb::arbor_exception("Unsupported cable-cell format version `" + version + "`")
{}


// Define s-expr makers for various types
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
    return slist("ion-internal-concentration"_symbol, s_expr(c.ion), c.value);
}
s_expr mksexp(const init_ext_concentration& c) {
    return slist("ion-external-concentration"_symbol, s_expr(c.ion), c.value);
}
s_expr mksexp(const init_reversal_potential& c) {
    return slist("ion-reversal-potential"_symbol, s_expr(c.ion), c.value);
}
s_expr mksexp(const ion_diffusivity& c) {
    return slist("ion-diffusivity"_symbol, s_expr(c.ion), c.value);
}
s_expr mksexp(const i_clamp& c) {
    std::vector<s_expr> evlps;
    std::transform(c.envelope.begin(), c.envelope.end(), std::back_inserter(evlps),
        [](const auto& x){return slist(x.t, x.amplitude);});
    return slist("current-clamp"_symbol, s_expr{"envelope"_symbol, slist_range(evlps)}, c.frequency, c.phase);
}
s_expr mksexp(const threshold_detector& d) {
    return slist("threshold-detector"_symbol, d.threshold);
}
s_expr mksexp(const mechanism_desc& d) {
    std::vector<s_expr> mech;
    mech.emplace_back(d.name());
    std::transform(d.values().begin(), d.values().end(), std::back_inserter(mech),
        [](const auto& x){return slist(s_expr(x.first), x.second);});
    return s_expr{"mechanism"_symbol, slist_range(mech)};
}
s_expr mksexp(const ion_reversal_potential_method& e) {
    return slist("ion-reversal-potential-method"_symbol, s_expr(e.ion), mksexp(e.method));
}
s_expr mksexp(const junction& j) {
    return slist("junction"_symbol, mksexp(j.mech));
}
s_expr mksexp(const synapse& j) {
    return slist("synapse"_symbol, mksexp(j.mech));
}
s_expr mksexp(const density& j) {
    return slist("density"_symbol, mksexp(j.mech));
}
s_expr mksexp(const voltage_process& j) {
    return slist("voltage-process"_symbol, mksexp(j.mech));
}
s_expr mksexp(const iexpr& j) {
    std::stringstream s;
    s << j;
    return parse_s_expr(s.str());
}
template <typename TaggedMech>
s_expr mksexp(const scaled_mechanism<TaggedMech>& j) {
    std::vector<s_expr> expressions;
    std::transform(j.scale_expr.begin(),
        j.scale_expr.end(),
        std::back_inserter(expressions),
        [](const auto& x) { return slist(s_expr(x.first), mksexp(x.second)); });
    return s_expr{"scaled-mechanism"_symbol, s_expr{mksexp(j.t_mech), slist_range(expressions)}};
}
s_expr mksexp(const mpoint& p) {
    return slist("point"_symbol, p.x, p.y, p.z, p.radius);
}
s_expr mksexp(const msegment& seg) {
    return slist("segment"_symbol, (int)seg.id, mksexp(seg.prox), mksexp(seg.dist), seg.tag);
}
// This can be removed once cv_policy is removed from the decor.
s_expr mksexp(const cv_policy& c) {
    std::stringstream s;
    s << c;
    return slist("cv-policy"_symbol, parse_s_expr(s.str()));
}
s_expr mksexp(const decor& d) {
    auto round_trip = [](auto& x) {
        std::stringstream s;
        s << x;
        return parse_s_expr(s.str());
    };
    std::vector<s_expr> decorations;
    for (const auto& p: d.defaults().serialize()) {
        decorations.push_back(std::visit([&](auto& x)
            { return slist("default"_symbol, mksexp(x)); }, p));
    }
    for (const auto& p: d.paintings()) {
        decorations.push_back(std::visit([&](auto& x)
            { return slist("paint"_symbol, round_trip(p.first), mksexp(x)); }, p.second));
    }
    for (const auto& p: d.placements()) {
        decorations.push_back(std::visit([&](auto& x)
            { return slist("place"_symbol, round_trip(std::get<0>(p)), mksexp(x), s_expr(std::get<2>(p))); }, std::get<1>(p)));
    }
    return {"decor"_symbol, slist_range(decorations)};
}
s_expr mksexp(const label_dict& dict) {
    auto round_trip = [](auto& x) {
        std::stringstream s;
        s << x;
        return parse_s_expr(s.str());
    };
    auto defs = slist();
    for (auto& r: dict.locsets()) {
        defs = s_expr(slist("locset-def"_symbol, s_expr(r.first), round_trip(r.second)), std::move(defs));
    }
    for (auto& r: dict.regions()) {
        defs = s_expr(slist("region-def"_symbol, s_expr(r.first), round_trip(r.second)), std::move(defs));
    }
    for (auto& r: dict.iexpressions()) {
        defs = s_expr(slist("iexpr-def"_symbol, s_expr(r.first), round_trip(r.second)), std::move(defs));
    }
    return {"label-dict"_symbol, std::move(defs)};
}
s_expr mksexp(const morphology& morph) {
    // s-expression representation of branch i in the morphology
    auto make_branch = [&morph](int i) {
        std::vector<s_expr> segments;
        const auto& segs = morph.branch_segments(i);
        std::transform(segs.begin(), segs.end(), std::back_inserter(segments),
            [](const auto& x){return mksexp(x);});
        return s_expr{"branch"_symbol, {i, {(int)morph.branch_parent(i), slist_range(segments)}}};
    };
    std::vector<s_expr> branches;
    for (msize_t i = 0; i < morph.num_branches(); ++i) {
        branches.emplace_back(make_branch(i));
    }
    return s_expr{"morphology"_symbol, slist_range(branches)};
}
s_expr mksexp(const meta_data& meta) {
    return slist("meta-data"_symbol, slist("version"_symbol, s_expr(meta.version)));
}

// Implement public facing s-expr writers
ARB_ARBORIO_API std::ostream& write_component(std::ostream& o, const decor& x, const meta_data& m) {
    if (m.version != acc_version()) {
        throw cableio_version_error(m.version);
    }
    return o << s_expr{"arbor-component"_symbol, slist(mksexp(m), mksexp(x))};
}
ARB_ARBORIO_API std::ostream& write_component(std::ostream& o, const label_dict& x, const meta_data& m) {
    if (m.version != acc_version()) {
        throw cableio_version_error(m.version);
    }
    return o << s_expr{"arbor-component"_symbol, slist(mksexp(m), mksexp(x))};
}
ARB_ARBORIO_API std::ostream& write_component(std::ostream& o, const morphology& x, const meta_data& m) {
    if (m.version != acc_version()) {
        throw cableio_version_error(m.version);
    }
    return o << s_expr{"arbor-component"_symbol, slist(mksexp(m), mksexp(x))};
}
ARB_ARBORIO_API std::ostream& write_component(std::ostream& o, const cable_cell& x, const meta_data& m) {
    if (m.version != acc_version()) {
        throw cableio_version_error(m.version);
    }
    auto cell = s_expr{"cable-cell"_symbol, slist(mksexp(x.morphology()), mksexp(x.labels()), mksexp(x.decorations()))};
    return o << s_expr{"arbor-component"_symbol, slist(mksexp(m), cell)};
}
ARB_ARBORIO_API std::ostream& write_component(std::ostream& o, const cable_cell_component& x) {
    if (x.meta.version != acc_version()) {
        throw cableio_version_error(x.meta.version);
    }
    std::visit([&](auto&& c){write_component(o, c, x.meta);}, x.component);
    return o;
}

// Anonymous namespace containing helper functions and types for parsing s-expr
namespace {

// Useful tuple aliases
using envelope_tuple = std::tuple<double,double>;
using pulse_tuple    = std::tuple<double,double,double>;
using param_tuple    = std::tuple<std::string,double>;
using branch_tuple   = std::tuple<int,int,std::vector<arb::msegment>>;
using version_tuple  = std::tuple<std::string>;

// Define makers for defaultables, paintables, placeables
#define ARBIO_DEFINE_SINGLE_ARG(name) arb::name make_##name(double val) { return arb::name{val}; }
#define ARBIO_DEFINE_DOUBLE_ARG(name) arb::name make_##name(const std::string& ion, double val) { return arb::name{ion, val};}

ARB_PP_FOREACH(ARBIO_DEFINE_SINGLE_ARG, init_membrane_potential, temperature_K, axial_resistivity, membrane_capacitance, threshold_detector)
ARB_PP_FOREACH(ARBIO_DEFINE_DOUBLE_ARG, init_int_concentration, init_ext_concentration, init_reversal_potential, ion_diffusivity)

std::vector<arb::i_clamp::envelope_point> make_envelope(const std::vector<std::variant<envelope_tuple>>& vec) {
    std::vector<arb::i_clamp::envelope_point> envlp;
    std::transform(vec.begin(), vec.end(), std::back_inserter(envlp),
        [](const auto& x){
            auto t = std::get<envelope_tuple>(x);
            return arb::i_clamp::envelope_point{std::get<0>(t), std::get<1>(t)};
        });
    return envlp;
}
arb::i_clamp make_i_clamp(const std::vector<arb::i_clamp::envelope_point>& envlp, double freq, double phase) {
    return arb::i_clamp(envlp, freq, phase);
}
pulse_tuple make_envelope_pulse(double delay, double duration, double amplitude) {
    return pulse_tuple{delay, duration, amplitude};
}
arb::i_clamp make_i_clamp_pulse(const pulse_tuple& p, double freq, double phase) {
    return arb::i_clamp::box(std::get<0>(p), std::get<1>(p), std::get<2>(p), freq, phase);
}
arb::cv_policy make_cv_policy(const cv_policy& p) {
    return p;
}
arb::ion_reversal_potential_method make_ion_reversal_potential_method(const std::string& ion, const arb::mechanism_desc& mech) {
    return ion_reversal_potential_method{ion, mech};
}
template <typename T>
T make_wrapped_mechanism(const arb::mechanism_desc& mech) {return T(mech);}
#undef ARBIO_DEFINE_SINGLE_ARG
#undef ARBIO_DEFINE_DOUBLE_ARG

// Define makers for placeable pairs, paintable pairs, defaultables and decors
using place_tuple = std::tuple<arb::locset, arb::placeable, std::string>;
using paint_pair = std::pair<arb::region, arb::paintable>;
place_tuple make_place(const locset& where, const placeable& what, const std::string& name) {
    return place_tuple{where, what, name};
}
paint_pair make_paint(const region& where, const paintable& what) {
    return paint_pair{where, what};
}
defaultable make_default(defaultable what) {
    return what;
}
decor make_decor(const std::vector<std::variant<place_tuple, paint_pair, defaultable>>& args) {
    decor d;
    for(const auto& a: args) {
        auto decor_visitor = arb::util::overload(
            [&](const place_tuple & p) { d.place(std::get<0>(p), std::get<1>(p), std::get<2>(p)); },
            [&](const paint_pair & p) { d.paint(p.first, p.second); },
            [&](const defaultable & p){ d.set_default(p); });
        std::visit(decor_visitor, a);
    }
    return d;
}

// Define maker for locset pairs, region pairs and label_dicts
using locset_pair = std::pair<std::string, locset>;
using region_pair = std::pair<std::string, region>;
using iexpr_pair = std::pair<std::string, iexpr>;
locset_pair make_locset_pair(std::string name, locset desc) {
    return locset_pair{std::move(name), std::move(desc)};
}
region_pair make_region_pair(std::string name, region desc) {
    return region_pair{std::move(name), std::move(desc)};
}
iexpr_pair make_iexpr_pair(std::string name, iexpr e) {
    return iexpr_pair{std::move(name), std::move(e)};
}
label_dict make_label_dict(const std::vector<std::variant<locset_pair, region_pair, iexpr_pair>>& args) {
    label_dict d;
    for(const auto& a: args) {
        std::visit([&](auto&& x){d.set(x.first, x.second);}, a);
    }
    return d;
}
// Define makers for mpoints and msegments and morphologies
arb::mpoint make_point(double x, double y, double z, double r) {
    return arb::mpoint{x, y, z, r};
}
arb::msegment make_segment(unsigned id, const arb::mpoint& prox, const arb::mpoint& dist, int tag) {
    return arb::msegment{id, prox, dist, tag};
}
morphology make_morphology(const std::vector<std::variant<branch_tuple>>& args) {
    segment_tree tree;
    std::vector<unsigned> branch_final_seg(args.size());
    std::vector<unsigned> branch_children(args.size(), 0);
    std::vector<std::pair<msegment, int>> segs;
    for (const auto& br: args) {
        auto b = std::get<branch_tuple>(br);
        auto b_id = std::get<0>(b);
        auto b_pid = std::get<1>(b);
        auto b_segments = std::get<2>(b);

        if (b_pid!=-1) branch_children[b_pid]++;

        auto s_pid = b_pid==-1? arb::mnpos: branch_final_seg[b_pid];
        for (const auto& s: b_segments) {
            segs.emplace_back(s, s_pid);
            s_pid = s.id;
        }
        branch_final_seg[b_id] = s_pid;
    }
    // A branch should have either 0 or >1 children.
    auto it = std::find_if(branch_children.begin(), branch_children.end(), [](unsigned i){return i==1;});
    if (it != branch_children.end()) {
        throw cableio_morphology_error(it - branch_children.begin());
    }

    // Append segments according to id order.
    std::sort(segs.begin(), segs.end(), [](const auto& lhs, const auto& rhs){return lhs.first.id < rhs.first.id;});
    for (const auto& [seg, s_pid]: segs) {
        tree.append(s_pid, seg.prox, seg.dist, seg.tag);
    }
    return morphology(tree);
}

// Define cable-cell maker
// Accepts the morphology, decor and label_dict arguments in any order as a vector
cable_cell make_cable_cell(const std::vector<std::variant<morphology, label_dict, decor>>& args) {
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
    return cable_cell(morpho, dec, dict);
}
version_tuple make_version(const std::string& v) {
    return version_tuple{v};
}
meta_data make_meta_data(const version_tuple& v) {
    return meta_data{std::get<0>(v)};
}
template <typename T>
cable_cell_component make_component(const meta_data& m, const T& d) {
    return cable_cell_component{m, d};
}

// Test whether a list of arguments passed as a std::vector<std::any> can be converted
// to a string followed by any number of std::pair<std::string, double>
struct mech_match {
    bool operator()(const std::vector<std::any>& args) const {
        // First argument is the mech name
        if (args.size() < 1) return false;
        if (!match<std::string>(args.front().type())) return false;

        // The rest of the arguments should be param_tuples
        for (auto it = args.begin()+1; it != args.end(); ++it) {
            if (!match<param_tuple>(it->type())) return false;
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
            auto p = eval_cast<param_tuple>(*it);
            mech.set(std::get<0>(p), std::get<1>(p));
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
// to a string followed by any number of std::pair<std::string, arb::iexpr>
template <typename TaggedMech>
struct scaled_mechanism_match {
    bool operator()(const std::vector<std::any>& args) const {
        // First argument is the mech name
        if (args.size() < 1) return false;
        if (!match<TaggedMech>(args.front().type())) return false;

        // The rest of the arguments should be tuples
        for (auto it = args.begin()+1; it != args.end(); ++it) {
            if (!match<std::tuple<std::string, arb::iexpr>>(it->type())) return false;
        }
        return true;
    }
};
// Create a mechanism_desc from a std::vector<std::any>.
template <typename TaggedMech>
struct scaled_mechanism_eval {
    arb::scaled_mechanism<TaggedMech> operator()(const std::vector<std::any>& args) {
        auto d = scaled_mechanism(eval_cast<TaggedMech>(args.front()));

        for (auto it = args.begin() + 1; it != args.end(); ++it) {
            auto p = eval_cast<std::tuple<std::string, arb::iexpr>>(*it);
            d.scale(std::move(std::get<0>(p)), std::move(std::get<1>(p)));
        }
        return d;
    }
};
// Wrap mech_match and mech_eval in an evaluator
template <typename TaggedMech>
struct make_scaled_mechanism_call {
    evaluator state;
    make_scaled_mechanism_call(const char* msg="scaled-mechanism"):
        state(scaled_mechanism_eval<TaggedMech>(), scaled_mechanism_match<TaggedMech>(), msg)
    {}
    operator evaluator() const {
        return state;
    }
};

// Test whether a list of arguments passed as a std::vector<std::any> can be converted
// to 2 integers followed by at least 1 msegment
struct branch_match {
    bool operator()(const std::vector<std::any>& args) const {
        if (args.size() < 2) return false;
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
    branch_tuple operator()(const std::vector<std::any>& args) {
        std::vector<msegment> segs;
        auto it = args.begin();
        auto id = eval_cast<int>(*it++);
        auto parent = eval_cast<int>(*it++);
        for (; it != args.end(); ++it) {
            segs.push_back(eval_cast<msegment>(*it));
        }
        return branch_tuple{id, parent, segs};
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
using eval_vec = std::vector<evaluator>;

// Parse s-expression into std::any given a function evaluation map and a tuple evaluation vector.
parse_hopefully<std::any> eval(const arb::s_expr&, const eval_map&, const eval_vec&);

parse_hopefully<std::vector<std::any>> eval_args(const s_expr& e, const eval_map& map, const eval_vec& vec) {
    if (!e) return {std::vector<std::any>{}};
    std::vector<std::any> args;
    for (auto& h: e) {
        if (auto arg=eval(h, map, vec)) {
            args.push_back(std::move(*arg));
        }
        else {
            return util::unexpected(std::move(arg.error()));
        }
    }
    return args;
}

parse_hopefully<std::any> eval(const s_expr& e, const eval_map& map, const eval_vec& vec) {
    if (e.is_atom()) {
        return eval_atom<cableio_parse_error>(e);
    }

    auto& hd = e.head();
    if (hd.is_atom()) {
        auto& atom = hd.atom();
        // If this is not a symbol, parse as a tuple
        if (atom.kind != tok::symbol) {
            auto args = eval_args(e, map, vec);
            if (!args) {
                return util::unexpected(args.error());
            }
            for (auto& e: vec) {
                if (e.match_args(*args)) { // found a match: evaluate and return.
                    return e.eval(*args);
                }
            }

            // Unable to find a match: try to return a helpful error message.
            const auto nc = vec.size();
            std::string msg = "No matches for found for unnamed tuple with "+std::to_string(args->size())+" arguments.\n"
                              "There are "+std::to_string(nc)+" potential candiates"+(nc?":":".");
            int count = 0;
            for (auto& e: vec) {
                msg += "\n  Candidate "+std::to_string(++count)+": "+e.message;
            }
            return util::unexpected(cableio_parse_error(msg, location(e)));
        };

        // Otherwise this must be a function evaluation, where head is the function name,
        // and tail is a list of arguments.
        // Evaluate the arguments, and return error state if an error occurred.
        auto args = eval_args(e.tail(), map, vec);
        if (!args) {
            return util::unexpected(args.error());
        }

        // Find all candidate functions that match the name of the function.
        auto& name = atom.spelling;
        auto matches = map.equal_range(name);

        // Search for a candidate that matches the argument list.
        for (auto i=matches.first; i!=matches.second; ++i) {
            if (i->second.match_args(*args)) { // found a match: evaluate and return.
                return i->second.eval(*args);
            }
        }

        // If it's not in the provided map, it could be a label expression
        if (auto l = parse_label_expression(e)) {
            if (match<region>(l->type())) return eval_cast<region>(l.value());
            if (match<locset>(l->type())) return eval_cast<locset>(l.value());
            if (match<iexpr>(l->type())) return eval_cast<iexpr>(l.value());
        }

        // Or it could be a cv-policy expression
        if (auto p = parse_cv_policy_expression(e)) return p.value();

        // Unable to find a match: try to return a helpful error message.
        const auto nc = std::distance(matches.first, matches.second);
        std::string msg = "No matches for found for "+name+" with "+std::to_string(args->size())+" arguments.\n"
                          "There are "+std::to_string(nc)+" potential candidates"+(nc?":":".");
        int count = 0;
        for (auto i=matches.first; i!=matches.second; ++i) {
            msg += "\n  Candidate "+std::to_string(++count)+": "+i->second.message;
        }
        return util::unexpected(cableio_parse_error(msg, location(e)));
    }
    return util::unexpected(cableio_parse_error("Expression is not integer, real expression of the form (op <args>) nor tuple of the form (e0 e1 ... en)", location(e)));
}

eval_map named_evals{
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
    {"ion-diffusivity", make_call<std::string, double>(make_ion_diffusivity,
        "'ion_diffusivity' with 2 arguments (ion:string val:real)")},
    {"ion-reversal-potential", make_call<std::string, double>(make_init_reversal_potential,
        "'ion_reversal_potential' with 2 arguments (ion:string val:real)")},
    {"envelope", make_arg_vec_call<envelope_tuple>(make_envelope,
        "'envelope' with one or more pairs of start time and amplitude (start:real amplitude:real)")},
    {"envelope-pulse", make_call<double, double, double>(make_envelope_pulse,
        "'envelope-pulse' with 3 arguments (delay:real duration:real amplitude:real)")},
    {"current-clamp", make_call<std::vector<arb::i_clamp::envelope_point>, double, double>(make_i_clamp,
        "'current-clamp' with 3 arguments (env:envelope freq:real phase:real)")},
    {"current-clamp", make_call<pulse_tuple, double, double>(make_i_clamp_pulse,
        "'current-clamp' with 3 arguments (env:envelope_pulse freq:real phase:real)")},
    {"threshold-detector", make_call<double>(make_threshold_detector,
        "'threshold-detector' with 1 argument (threshold:real)")},
    {"mechanism", make_mech_call("'mechanism' with a name argument, and 0 or more parameter settings"
        "(name:string (param:string val:real))")},
    {"ion-reversal-potential-method", make_call<std::string, arb::mechanism_desc>( make_ion_reversal_potential_method,
        "'ion-reversal-potential-method' with 2 arguments (ion:string mech:mechanism)")},
    {"cv-policy", make_call<cv_policy>(make_cv_policy,
        "'cv-policy' with 1 argument (p:policy)")},
    {"junction", make_call<arb::mechanism_desc>(make_wrapped_mechanism<junction>, "'junction' with 1 argumnet (m: mechanism)")},
    {"synapse",  make_call<arb::mechanism_desc>(make_wrapped_mechanism<synapse>, "'synapse' with 1 argumnet (m: mechanism)")},
    {"density",  make_call<arb::mechanism_desc>(make_wrapped_mechanism<density>, "'density' with 1 argumnet (m: mechanism)")},
    {"voltage-process",  make_call<arb::mechanism_desc>(make_wrapped_mechanism<voltage_process>, "'voltage-process' with 1 argumnet (m: mechanism)")},
    {"scaled-mechanism", make_scaled_mechanism_call<arb::density>("'scaled_mechanism' with a density argument, and 0 or more parameter scaling expressions"
        "(d:density (param:string val:iexpr))")},
    {"place", make_call<locset, i_clamp, std::string>(make_place, "'place' with 3 arguments (ls:locset c:current-clamp name:string)")},
    {"place", make_call<locset, threshold_detector, std::string>(make_place, "'place' with 3 arguments (ls:locset t:threshold-detector name:string)")},
    {"place", make_call<locset, junction, std::string>(make_place, "'place' with 3 arguments (ls:locset gj:junction name:string)")},
    {"place", make_call<locset, synapse, std::string>(make_place, "'place' with 3 arguments (ls:locset mech:synapse name:string)")},

    {"paint", make_call<region, init_membrane_potential>(make_paint, "'paint' with 2 arguments (reg:region v:membrane-potential)")},
    {"paint", make_call<region, temperature_K>(make_paint, "'paint' with 2 arguments (reg:region v:temperature-kelvin)")},
    {"paint", make_call<region, membrane_capacitance>(make_paint, "'paint' with 2 arguments (reg:region v:membrane-capacitance)")},
    {"paint", make_call<region, axial_resistivity>(make_paint, "'paint' with 2 arguments (reg:region v:axial-resistivity)")},
    {"paint", make_call<region, init_int_concentration>(make_paint, "'paint' with 2 arguments (reg:region v:ion-internal-concentration)")},
    {"paint", make_call<region, init_ext_concentration>(make_paint, "'paint' with 2 arguments (reg:region v:ion-external-concentration)")},
    {"paint", make_call<region, ion_diffusivity>(make_paint, "'paint' with 2 arguments (reg:region v:ion-diffusivity)")},
    {"paint", make_call<region, init_reversal_potential>(make_paint, "'paint' with 2 arguments (reg:region v:ion-reversal-potential)")},
    {"paint", make_call<region, density>(make_paint, "'paint' with 2 arguments (reg:region v:density)")},
    {"paint", make_call<region, scaled_mechanism<density>>(make_paint, "'paint' with 2 arguments (reg:region v:scaled-mechanism)")},

    {"default", make_call<init_membrane_potential>(make_default, "'default' with 1 argument (v:membrane-potential)")},
    {"default", make_call<temperature_K>(make_default, "'default' with 1 argument (v:temperature-kelvin)")},
    {"default", make_call<membrane_capacitance>(make_default, "'default' with 1 argument (v:membrane-capacitance)")},
    {"default", make_call<axial_resistivity>(make_default, "'default' with 1 argument (v:axial-resistivity)")},
    {"default", make_call<init_int_concentration>(make_default, "'default' with 1 argument (v:ion-internal-concentration)")},
    {"default", make_call<init_ext_concentration>(make_default, "'default' with 1 argument (v:ion-external-concentration)")},
    {"default", make_call<ion_diffusivity>(make_default, "'default' with 1 argument (v:ion-diffusivity)")},
    {"default", make_call<init_reversal_potential>(make_default, "'default' with 1 argument (v:ion-reversal-potential)")},
    {"default", make_call<ion_reversal_potential_method>(make_default, "'default' with 1 argument (v:ion-reversal-potential-method)")},
    {"default", make_call<cv_policy>(make_default, "'default' with 1 argument (v:cv-policy)")},

    {"locset-def", make_call<std::string, locset>(make_locset_pair,
        "'locset-def' with 2 arguments (name:string ls:locset)")},
    {"region-def", make_call<std::string, region>(make_region_pair,
        "'region-def' with 2 arguments (name:string reg:region)")},
    {"iexpr-def", make_call<std::string, iexpr>(make_iexpr_pair,
        "'iexpr-def' with 2 arguments (name:string e:iexpr)")},

    {"point",   make_call<double, double, double, double>(make_point,
         "'point' with 4 arguments (x:real y:real z:real radius:real)")},
    {"segment", make_call<int, mpoint, mpoint, int>(make_segment,
        "'segment' with 4 arguments (parent:int prox:point dist:point tag:int)")},
    {"branch",  make_branch_call(
        "'branch' with 2 integers and 1 or more segment arguments (id:int parent:int s0:segment s1:segment ..)")},

    {"decor", make_arg_vec_call<place_tuple, paint_pair, defaultable>(make_decor,
        "'decor' with 1 or more `paint`, `place` or `default` arguments")},
    {"label-dict", make_arg_vec_call<locset_pair, region_pair, iexpr_pair>(make_label_dict,
        "'label-dict' with 1 or more `locset-def` or `region-def` or `iexpr-def` arguments")},
    {"morphology", make_arg_vec_call<branch_tuple>(make_morphology,
        "'morphology' 1 or more `branch` arguments")},

    {"cable-cell", make_unordered_call<morphology, label_dict, decor>(make_cable_cell,
        "'cable-cell' with 3 arguments: `morphology`, `label-dict`, and `decor` in any order")},

    {"version", make_call<std::string>(make_version, "'version' with one argment (val:std::string)")},
    {"meta-data", make_call<version_tuple>(make_meta_data, "'meta-data' with one argument (v:version)")},

    { "arbor-component", make_call<meta_data, decor>(make_component<decor>, "'arbor-component' with 2 arguments (m:meta_data p:decor)")},
    { "arbor-component", make_call<meta_data, label_dict>(make_component<label_dict>, "'arbor-component' with 2 arguments (m:meta_data p:label_dict)")},
    { "arbor-component", make_call<meta_data, morphology>(make_component<morphology>, "'arbor-component' with 2 arguments (m:meta_data p:morphology)")},
    { "arbor-component", make_call<meta_data, cable_cell>(make_component<cable_cell>, "'arbor-component' with 2 arguments (m:meta_data p:cable_cell)")}
};

eval_vec unnamed_evals{
    make_call<std::string, double>(std::make_tuple<std::string, double>, "tuple<std::string, double>"),
    make_call<double, double>(std::make_tuple<double, double>, "tuple<double, double>"),
    make_call<std::string, arb::iexpr>(std::make_tuple<std::string, arb::iexpr>, "tuple<std::string, arb::iexpr>"),
};

inline parse_hopefully<std::any> parse(const arb::s_expr& s) {
    return eval(s, named_evals, unnamed_evals);
}

ARB_ARBORIO_API parse_hopefully<std::any> parse_expression(const std::string& s) {
    return parse(parse_s_expr(s));
}

// Read s-expr
ARB_ARBORIO_API parse_hopefully<cable_cell_component> parse_component(const std::string& s) {
    auto sexp = parse_s_expr(s);
    auto try_parse = parse(sexp);
    if (!try_parse) {
        return util::unexpected(cableio_parse_error(try_parse.error()));
    }
    if (!match<cable_cell_component>(try_parse.value().type())) {
        return util::unexpected(cableio_parse_error("Expected arbor-component", location(sexp)));
    }
    auto comp = eval_cast<cable_cell_component>(try_parse.value());
    if (comp.meta.version != acc_version()) {
        return util::unexpected(cableio_parse_error("Unsupported cable-cell format version "+ comp.meta.version, location(sexp)));
    }
    return comp;
};

ARB_ARBORIO_API parse_hopefully<cable_cell_component> parse_component(std::istream& s) {
    return parse_component(std::string(std::istreambuf_iterator<char>(s), {}));
}
} // namespace arborio
