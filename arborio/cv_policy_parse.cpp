#include <any>
#include <limits>
#include <optional>

#include <arborio/label_parse.hpp>
#include <arborio/cv_policy_parse.hpp>

#include <arbor/arbexcept.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/s_expr.hpp>
#include <arbor/util/expected.hpp>


#include "parse_helpers.hpp"

namespace arborio {

cv_policy_parse_error::cv_policy_parse_error(const std::string& msg, const arb::src_location& loc):
    arb::arbor_exception(concat("error in CV policy description: ", msg," at :", loc.line, ":", loc.column))
{}

cv_policy_parse_error::cv_policy_parse_error(const std::string& msg):
    arb::arbor_exception(concat("error in CV policy description: ", msg))
{}

namespace {

template<typename T> using parse_hopefully = arb::util::expected<T, cv_policy_parse_error>;

std::unordered_multimap<std::string, evaluator>
eval_map {{"default",
           make_call<>([] () { return arb::cv_policy{arb::default_cv_policy()}; },
                       "'default' with no arguments")},
          {"every-segment",
           make_call<>([] () { return arb::cv_policy{arb::cv_policy_every_segment()}; },
                       "'every-segment' with no arguments")},
          {"every-segment",
           make_call<region>([] (const region& r) { return arb::cv_policy{arb::cv_policy_every_segment(r) }; },
                             "'every-segment' with one argument (every-segment (reg:region))")},
          {"fixed-per-branch",
           make_call<int>([] (int i) { return arb::cv_policy{arb::cv_policy_fixed_per_branch(i) }; },
                          "'every-segment' with one argument (fixed-per-branch (count:int))")},
          {"fixed-per-branch",
           make_call<int, region>([] (int i, const region& r) { return arb::cv_policy{arb::cv_policy_fixed_per_branch(i, r) }; },
                                  "'every-segment' with two arguments (fixed-per-branch (count:int) (reg:region))")},
          {"fixed-per-branch",
           make_call<int, region, int>([] (int i, const region& r, int f) { return arb::cv_policy{arb::cv_policy_fixed_per_branch(i, r, f) }; },
                                       "'fixed-per-branch' with three arguments (fixed-per-branch (count:int) (reg:region) (flags:int))")},
          {"max-extent",
           make_call<double>([] (double i) { return arb::cv_policy{arb::cv_policy_max_extent(i) }; },
                             "'max-extent' with one argument (max-extent (length:double))")},
          {"max-extent",
           make_call<double, region>([] (double i, const region& r) { return arb::cv_policy{arb::cv_policy_max_extent(i, r) }; },
                                     "'max-extent' with two arguments (max-extent (length:double) (reg:region))")},
          {"max-extent",
           make_call<double, region, int>([] (double i, const region& r, int f) { return arb::cv_policy{arb::cv_policy_max_extent(i, r, f) }; },
                                          "'max-extent' with three arguments (max-extent (length:double) (reg:region) (flags:int))")},
          {"single",
           make_call<>([] () { return arb::cv_policy{arb::cv_policy_single()}; },
                       "'single' with no arguments")},
          {"single",
           make_call<region>([] (const region& r) { return arb::cv_policy{arb::cv_policy_single(r) }; },
                             "'single' with one argument (single (reg:region))")},
          {"explicit",
           make_call<locset>([] (const locset& l) { return arb::cv_policy{arb::cv_policy_explicit(l) }; },
                             "'explicit' with one argument (explicit (ls:locset))")},
          {"explicit",
           make_call<locset, region>([] (const locset& l, const region& r) { return arb::cv_policy{arb::cv_policy_explicit(l, r) }; },
                                     "'explicit' with two arguments (explicit (ls:locset) (reg:region))")},
          {"join",
           make_fold<cv_policy>([](cv_policy l, cv_policy r) { return l + r; },
                                "'join' with at least 2 arguments: (join cv_policy cv_policy ...)")},
          {"replace",
           make_fold<cv_policy>([](cv_policy l, cv_policy r) { return l | r; },
                                "'replace' with at least 2 arguments: (replace cv_policy cv_policy ...)")},
};

parse_hopefully<std::any> eval(const s_expr& e);

parse_hopefully<std::vector<std::any>> eval_args(const s_expr& e) {
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
        if (t==typeid(int))       return "integer";
        if (t==typeid(double))    return "real";
        if (t==typeid(region))    return "region";
        if (t==typeid(locset))    return "locset";
        if (t==typeid(cv_policy)) return "cv_policy";
        return "unknown";
    };

    const auto nargs = args.size();
    std::string msg = concat("'", name, "' with ", nargs, " argument", nargs!=1u ? "s" : "", ":");
    if (nargs) {
        msg += " (";
        bool append_sep = false;
        for (auto& a: args) {
            if (append_sep) {
                msg += " ";
            }
            msg += type_string(a.type());
            append_sep = true;
        }
        msg += ")";
    }
    return msg;
}

// Evaluate an s expression.
// On success the result is wrapped in std::any, where the result is one of:
//      int:            an integer atom
//      double:         a real atom
//      cv_policy:      a discretization policy expression
//
// If there invalid input is detected, hopefully return value contains
// a cv_policy_error_state with an error string and location.
//
// If there was an unexpected/fatal error, an exception will be thrown.
parse_hopefully<std::any> eval(const s_expr& e) {
    if (e.is_atom()) {
        return eval_atom<cv_policy_parse_error>(e);
    }

    if (e.head().is_atom()) {
        // This must be a function evaluation, where head is the function name, and
        // tail is a list of arguments.

        // Evaluate the arguments, and return error state if an error occurred.
        auto args = eval_args(e.tail());
        if (!args) {
            return args.error();
        }

        // Find all candidate functions that match the name of the function.
        auto& name = e.head().atom().spelling;
        auto matches = eval_map.equal_range(name);

        // if no matches found, maybe this is a morphology expression?
        if (matches.first == matches.second) {
            auto lbl = parse_label_expression(e);
            if (lbl.has_value()) {
                return { lbl.value() };
            } else {
                return util::unexpected(cv_policy_parse_error(lbl.error().what(), location(e)));
            }
        } else {
            // Search for a candidate that matches the argument list.
            for (auto i=matches.first; i!=matches.second; ++i) {
                if (i->second.match_args(*args)) { // found a match: evaluate and return.
                    return i->second.eval(*args);
                }
            }

            // Unable to find a match: try to return a helpful error message.
            const auto nc = std::distance(matches.first, matches.second);
            auto msg = concat("No matches for ", eval_description(name.c_str(), *args), "\n  There are ", nc, " potential candiates", nc ? ":" : ".");
            int count = 0;
            for (auto i=matches.first; i!=matches.second; ++i) {
                msg += concat("\n  Candidate ", ++count, ": ", i->second.message);
            }

            return util::unexpected(cv_policy_parse_error(msg, location(e)));
        }
    }

    return util::unexpected(cv_policy_parse_error(
                                concat("'", e, "' is not either integer, real expression of the form (op <args>)"),
                                location(e)));
}
}

ARB_ARBORIO_API parse_cv_policy_hopefully parse_cv_policy_expression(const arb::s_expr& s) {
    if (auto e = eval(s)) {
        if (e->type() == typeid(cv_policy)) {
            return {std::move(std::any_cast<cv_policy&>(*e))};
        }
        return util::unexpected(
                cv_policy_parse_error(concat("Invalid description: '", s, "' is not a valid CV policy expression.")));
    }
    else {
        return util::unexpected(cv_policy_parse_error(std::string() + e.error().what()));
    }
}
ARB_ARBORIO_API parse_cv_policy_hopefully parse_cv_policy_expression(const std::string& s) {
    return parse_cv_policy_expression(parse_s_expr(s));
}
} // namespace arb
