#include <any>
#include <limits>

#include <arborio/label_parse.hpp>

#include <arbor/arbexcept.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/iexpr.hpp>

#include <arbor/util/expected.hpp>

#include "parse_helpers.hpp"

namespace arborio {

label_parse_error::label_parse_error(const std::string& msg, const arb::src_location& loc):
    arb::arbor_exception(concat("error in label description: ", msg," at :", loc.line, ":", loc.column))
{}


namespace {

std::unordered_multimap<std::string, evaluator> eval_map {
    // Functions that return regions
    {"region-nil", make_call<>(arb::reg::nil,
                "'region-nil' with 0 arguments")},
    {"all", make_call<>(arb::reg::all,
                "'all' with 0 arguments")},
    {"tag", make_call<int>(arb::reg::tagged,
                "'tag' with 1 argment: (tag_id:integer)")},
    {"segment", make_call<int>(arb::reg::segment,
                    "'segment' with 1 argment: (segment_id:integer)")},
    {"branch", make_call<int>(arb::reg::branch,
                   "'branch' with 1 argument: (branch_id:integer)")},
    {"cable", make_call<int, double, double>(arb::reg::cable,
                  "'cable' with 3 arguments: (branch_id:integer prox:real dist:real)")},
    {"region", make_call<std::string>(arb::reg::named,
                   "'region' with 1 argument: (name:string)")},
    {"distal-interval", make_call<arb::locset, double>(arb::reg::distal_interval,
                            "'distal-interval' with 2 arguments: (start:locset extent:real)")},
    {"distal-interval", make_call<arb::locset>(
                            [](arb::locset ls){return arb::reg::distal_interval(std::move(ls), std::numeric_limits<double>::max());},
                            "'distal-interval' with 1 argument: (start:locset)")},
    {"proximal-interval", make_call<arb::locset, double>(arb::reg::proximal_interval,
                              "'proximal-interval' with 2 arguments: (start:locset extent:real)")},
    {"proximal-interval", make_call<arb::locset>(
                              [](arb::locset ls){return arb::reg::proximal_interval(std::move(ls), std::numeric_limits<double>::max());},
                              "'proximal_interval' with 1 argument: (start:locset)")},
    {"complete", make_call<arb::region>(arb::reg::complete,
                     "'complete' with 1 argment: (reg:region)")},
    {"radius-lt", make_call<arb::region, double>(arb::reg::radius_lt,
                      "'radius-lt' with 2 arguments: (reg:region radius:real)")},
    {"radius-le", make_call<arb::region, double>(arb::reg::radius_le,
                      "'radius-le' with 2 arguments: (reg:region radius:real)")},
    {"radius-gt", make_call<arb::region, double>(arb::reg::radius_gt,
                      "'radius-gt' with 2 arguments: (reg:region radius:real)")},
    {"radius-ge", make_call<arb::region, double>(arb::reg::radius_ge,
                      "'radius-ge' with 2 arguments: (reg:region radius:real)")},
    {"z-dist-from-root-lt", make_call<double>(arb::reg::z_dist_from_root_lt,
                                "'z-dist-from-root-lt' with 1 arguments: (distance:real)")},
    {"z-dist-from-root-le", make_call<double>(arb::reg::z_dist_from_root_le,
                                "'z-dist-from-root-le' with 1 arguments: (distance:real)")},
    {"z-dist-from-root-gt", make_call<double>(arb::reg::z_dist_from_root_gt,
                                "'z-dist-from-root-gt' with 1 arguments: (distance:real)")},
    {"z-dist-from-root-ge", make_call<double>(arb::reg::z_dist_from_root_ge,
                                "'z-dist-from-root-ge' with 1 arguments: (distance:real)")},
    {"complement", make_call<arb::region>(arb::complement,
                       "'complement' with 1 argment: (reg:region)")},
    {"difference", make_call<arb::region, arb::region>(arb::difference,
                       "'difference' with 2 argments: (reg:region, reg:region)")},
    {"join", make_fold<arb::region>(static_cast<arb::region(*)(arb::region, arb::region)>(arb::join),
                 "'join' with at least 2 arguments: (region region [...region])")},
    {"intersect", make_fold<arb::region>(static_cast<arb::region(*)(arb::region, arb::region)>(arb::intersect),
                      "'intersect' with at least 2 arguments: (region region [...region])")},

    // Functions that return locsets
    {"locset-nil", make_call<>(arb::ls::nil,
                "'locset-nil' with 0 arguments")},
    {"root", make_call<>(arb::ls::root,
                 "'root' with 0 arguments")},
    {"location", make_call<int, double>([](int bid, double pos){return arb::ls::location(arb::msize_t(bid), pos);},
                     "'location' with 2 arguments: (branch_id:integer position:real)")},
    {"terminal", make_call<>(arb::ls::terminal,
                     "'terminal' with 0 arguments")},
    {"distal", make_call<arb::region>(arb::ls::most_distal,
                   "'distal' with 1 argument: (reg:region)")},
    {"proximal", make_call<arb::region>(arb::ls::most_proximal,
                     "'proximal' with 1 argument: (reg:region)")},
    {"distal-translate", make_call<arb::locset, double>(arb::ls::distal_translate,
                     "'distal-translate' with 2 arguments: (ls:locset distance:real)")},
    {"proximal-translate", make_call<arb::locset, double>(arb::ls::proximal_translate,
                     "'proximal-translate' with 2 arguments: (ls:locset distance:real)")},
    {"uniform", make_call<arb::region, int, int, int>(arb::ls::uniform,
                    "'uniform' with 4 arguments: (reg:region, first:int, last:int, seed:int)")},
    {"on-branches", make_call<double>(arb::ls::on_branches,
                        "'on-branches' with 1 argument: (pos:double)")},
    {"on-components", make_call<double, arb::region>(arb::ls::on_components,
                          "'on-components' with 2 arguments: (pos:double, reg:region)")},
    {"boundary", make_call<arb::region>(arb::ls::boundary,
                     "'boundary' with 1 argument: (reg:region)")},
    {"cboundary", make_call<arb::region>(arb::ls::cboundary,
                      "'cboundary' with 1 argument: (reg:region)")},
    {"segment-boundaries", make_call<>(arb::ls::segment_boundaries,
                               "'segment-boundaries' with 0 arguments")},
    {"support", make_call<arb::locset>(arb::ls::support,
                    "'support' with 1 argument (ls:locset)")},
    {"locset", make_call<std::string>(arb::ls::named,
                   "'locset' with 1 argument: (name:string)")},
    {"restrict", make_call<arb::locset, arb::region>(arb::ls::restrict,
                     "'restrict' with 2 arguments: (ls:locset, reg:region)")},
    {"join", make_fold<arb::locset>(static_cast<arb::locset(*)(arb::locset, arb::locset)>(arb::join),
                 "'join' with at least 2 arguments: (locset locset [...locset])")},
    {"sum", make_fold<arb::locset>(static_cast<arb::locset(*)(arb::locset, arb::locset)>(arb::sum),
                "'sum' with at least 2 arguments: (locset locset [...locset])")},


    // iexpr
    {"iexpr", make_call<std::string>(arb::iexpr::named, "iexpr with 1 argument: (value:string)")},

    {"scalar", make_call<double>(arb::iexpr::scalar, "iexpr with 1 argument: (value:double)")},

    {"pi", make_call<>(arb::iexpr::pi, "iexpr with no argument")},

    {"distance", make_call<double, arb::locset>(static_cast<arb::iexpr(*)(double, arb::locset)>(arb::iexpr::distance),
            "iexpr with 2 arguments: (scale:double, loc:locset)")},
    {"distance", make_call<arb::locset>(static_cast<arb::iexpr(*)(arb::locset)>(arb::iexpr::distance),
            "iexpr with 1 argument: (loc:locset)")},
    {"distance", make_call<double, arb::region>(static_cast<arb::iexpr(*)(double, arb::region)>(arb::iexpr::distance),
            "iexpr with 2 arguments: (scale:double, reg:region)")},
    {"distance", make_call<arb::region>(static_cast<arb::iexpr(*)(arb::region)>(arb::iexpr::distance),
            "iexpr with 1 argument: (reg:region)")},

    {"proximal-distance", make_call<double, arb::locset>(static_cast<arb::iexpr(*)(double, arb::locset)>(arb::iexpr::proximal_distance),
            "iexpr with 2 arguments: (scale:double, loc:locset)")},
    {"proximal-distance", make_call<arb::locset>(static_cast<arb::iexpr(*)(arb::locset)>(arb::iexpr::proximal_distance),
            "iexpr with 1 argument: (loc:locset)")},
    {"proximal-distance", make_call<double, arb::region>(static_cast<arb::iexpr(*)(double, arb::region)>(arb::iexpr::proximal_distance),
            "iexpr with 2 arguments: (scale:double, reg:region)")},
    {"proximal-distance", make_call<arb::region>(static_cast<arb::iexpr(*)(arb::region)>(arb::iexpr::proximal_distance),
            "iexpr with 1 arguments: (reg:region)")},

    {"distal-distance", make_call<double, arb::locset>(static_cast<arb::iexpr(*)(double, arb::locset)>(arb::iexpr::distal_distance),
            "iexpr with 2 arguments: (scale:double, loc:locset)")},
    {"distal-distance", make_call<arb::locset>(static_cast<arb::iexpr(*)(arb::locset)>(arb::iexpr::distal_distance),
            "iexpr with 1 argument: (loc:locset)")},
    {"distal-distance", make_call<double, arb::region>(static_cast<arb::iexpr(*)(double, arb::region)>(arb::iexpr::distal_distance),
            "iexpr with 2 arguments: (scale:double, reg:region)")},
    {"distal-distance", make_call<arb::region>(static_cast<arb::iexpr(*)(arb::region)>(arb::iexpr::distal_distance),
            "iexpr with 1 argument: (reg:region)")},

    {"interpolation", make_call<double, arb::locset, double, locset>(static_cast<arb::iexpr(*)(double, arb::locset, double, arb::locset)>(arb::iexpr::interpolation),
            "iexpr with 4 arguments: (prox_value:double, prox_list:locset, dist_value:double, dist_list:locset)")},
    {"interpolation", make_call<double, arb::region, double, region>(static_cast<arb::iexpr(*)(double, arb::region, double, arb::region)>(arb::iexpr::interpolation),
            "iexpr with 4 arguments: (prox_value:double, prox_list:region, dist_value:double, dist_list:region)")},

    {"radius", make_call<double>(static_cast<arb::iexpr(*)(double)>(arb::iexpr::radius), "iexpr with 1 argument: (value:double)")},
    {"radius", make_call<>(static_cast<arb::iexpr(*)()>(arb::iexpr::radius), "iexpr with no argument")},

    {"diameter", make_call<double>(static_cast<arb::iexpr(*)(double)>(arb::iexpr::diameter), "iexpr with 1 argument: (value:double)")},
    {"diameter", make_call<>(static_cast<arb::iexpr(*)()>(arb::iexpr::diameter), "iexpr with no argument")},

    {"exp", make_call<arb::iexpr>(arb::iexpr::exp, "iexpr with 1 argument: (value:iexpr)")},
    {"exp", make_call<double>(arb::iexpr::exp, "iexpr with 1 argument: (value:double)")},

    {"step_right", make_call<arb::iexpr>(arb::iexpr::step_right, "iexpr with 1 argument: (value:iexpr)")},
    {"step_right", make_call<double>(arb::iexpr::step_right, "iexpr with 1 argument: (value:double)")},

    {"step_left", make_call<arb::iexpr>(arb::iexpr::step_left, "iexpr with 1 argument: (value:iexpr)")},
    {"step_left", make_call<double>(arb::iexpr::step_left, "iexpr with 1 argument: (value:double)")},

    {"step", make_call<arb::iexpr>(arb::iexpr::step, "iexpr with 1 argument: (value:iexpr)")},
    {"step", make_call<double>(arb::iexpr::step, "iexpr with 1 argument: (value:double)")},

    {"log", make_call<arb::iexpr>(arb::iexpr::log, "iexpr with 1 argument: (value:iexpr)")},
    {"log", make_call<double>(arb::iexpr::log, "iexpr with 1 argument: (value:double)")},

    {"add", make_conversion_fold<arb::iexpr, arb::iexpr, double>(arb::iexpr::add, "iexpr with at least 2 arguments: ((iexpr | double) (iexpr | double) [...(iexpr | double)])")},

    {"sub", make_conversion_fold<arb::iexpr, arb::iexpr, double>(arb::iexpr::sub, "iexpr with at least 2 arguments: ((iexpr | double) (iexpr | double) [...(iexpr | double)])")},

    {"mul", make_conversion_fold<arb::iexpr, arb::iexpr, double>(arb::iexpr::mul, "iexpr with at least 2 arguments: ((iexpr | double) (iexpr | double) [...(iexpr | double)])")},

    {"div", make_conversion_fold<arb::iexpr, arb::iexpr, double>(arb::iexpr::div, "iexpr with at least 2 arguments: ((iexpr | double) (iexpr | double) [...(iexpr | double)])")},
};

parse_label_hopefully<std::any> eval(const s_expr& e);

parse_label_hopefully<std::vector<std::any>> eval_args(const s_expr& e) {
    if (!e) return {std::vector<std::any>{}}; // empty argument list
    std::vector<std::any> args;
    for (auto& h: e) {
        if (auto arg=eval(h)) {
            args.push_back(std::move(*arg));
        }
        else {
            return util::unexpected(std::move(arg.error()));
        }
    }
    return args;
}

// Generate a string description of a function evaluation of the form:
// Example output:
//  'foo' with 1 argument: (real)
//  'bar' with 0 arguments
//  'cat' with 3 arguments: (locset region integer)
// Where 'foo', 'bar' and 'cat' are the name of the function, and the
// types (integer, real, region, locset) are inferred from the arguments.
std::string eval_description(const char* name, const std::vector<std::any>& args) {
    auto type_string = [](const std::type_info& t) -> const char* {
        if (t==typeid(int))         return "integer";
        if (t==typeid(double))      return "real";
        if (t==typeid(arb::region)) return "region";
        if (t==typeid(arb::locset)) return "locset";
        return "unknown";
    };

    const auto nargs = args.size();
    std::string msg = concat("'", name, "' with ", nargs, "argument", nargs!=1u?"s:" : ":");
    if (nargs) {
        msg += " (";
        bool first = true;
        for (auto& a: args) {
            msg += concat(first?"":" ", type_string(a.type()));
            first = false;
        }
        msg += ")";
    }
    return msg;
}

// Evaluate an s expression.
// On success the result is wrapped in std::any, where the result is one of:
//      int         : an integer atom
//      double      : a real atom
//      std::string : a string atom: to be treated as a label
//      arb::region : a region
//      arb::locset : a locset
//
// If there invalid input is detected, hopefully return value contains
// a label_error_state with an error string and location.
//
// If there was an unexpected/fatal error, an exception will be thrown.
parse_label_hopefully<std::any> eval(const s_expr& e) {
    if (e.is_atom()) {
        return eval_atom<label_parse_error>(e);
    }
    if (e.head().is_atom()) {
        // This must be a function evaluation, where head is the function name, and
        // tail is a list of arguments.

        // Evaluate the arguments, and return error state if an error occurred.
        auto args = eval_args(e.tail());
        if (!args) {
            return util::unexpected(args.error());
        }

        // Find all candidate functions that match the name of the function.
        auto& name = e.head().atom().spelling;
        auto matches = eval_map.equal_range(name);
        // Search for a candidate that matches the argument list.
        for (auto i=matches.first; i!=matches.second; ++i) {
            if (i->second.match_args(*args)) { // found a match: evaluate and return.
                return i->second.eval(*args);
            }
        }

        // Unable to find a match: try to return a helpful error message.
        const auto nc = std::distance(matches.first, matches.second);
        auto msg = concat("No matches for ", eval_description(name.c_str(), *args), "\n  There are ", nc, " potential candidates", nc?":":".");
        int count = 0;
        for (auto i=matches.first; i!=matches.second; ++i) {
            msg += concat("\n  Candidate ", ++count, "  ", i->second.message);
        }
        return util::unexpected(label_parse_error(msg, location(e)));
    }

    return util::unexpected(label_parse_error(
                                concat("'", e, "' is not either integer, real expression of the form (op <args>)"),
                                location(e)));
}

} // namespace

ARB_ARBORIO_API parse_label_hopefully<std::any> parse_label_expression(const std::string& e) {
    return eval(parse_s_expr(e));
}
ARB_ARBORIO_API parse_label_hopefully<std::any> parse_label_expression(const s_expr& s) {
    return eval(s);
}

ARB_ARBORIO_API parse_label_hopefully<arb::region> parse_region_expression(const std::string& s) {
    if (auto e = eval(parse_s_expr(s))) {
        if (e->type() == typeid(region)) {
            return {std::move(std::any_cast<region&>(*e))};
        }
        if (e->type() == typeid(std::string)) {
            return {reg::named(std::move(std::any_cast<std::string&>(*e)))};
        }
        return util::unexpected(
                label_parse_error(
                    concat("Invalid region description: '", s ,"' is neither a valid region expression or region label string.")));
    }
    else {
        return util::unexpected(label_parse_error(std::string()+e.error().what()));
    }
}

ARB_ARBORIO_API parse_label_hopefully<arb::locset> parse_locset_expression(const std::string& s) {
    if (auto e = eval(parse_s_expr(s))) {
        if (e->type() == typeid(locset)) {
            return {std::move(std::any_cast<locset&>(*e))};
        }
        if (e->type() == typeid(std::string)) {
            return {ls::named(std::move(std::any_cast<std::string&>(*e)))};
        }
        return util::unexpected(
            label_parse_error(
                    concat("Invalid region description: '", s ,"' is neither a valid locset expression or locset label string.")));
    }
    else {
        return util::unexpected(label_parse_error(std::string()+e.error().what()));
    }
}

parse_label_hopefully<arb::iexpr> parse_iexpr_expression(const std::string& s) {
    if (auto e = eval(parse_s_expr(s))) {
        if (e->type() == typeid(iexpr)) {
            return {std::move(std::any_cast<iexpr&>(*e))};
        }
        return util::unexpected(
                label_parse_error(
                    concat("Invalid iexpr description: '", s)));
    }
    else {
        return util::unexpected(label_parse_error(std::string()+e.error().what()));
    }
}

} // namespace arborio
