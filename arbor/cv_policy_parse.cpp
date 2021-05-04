#include <any>
#include <limits>
#include <optional>

#include <arbor/arbexcept.hpp>
#include <arbor/cv_policy_parse.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/s_expr.hpp>
#include <arbor/util/expected.hpp>
#include <arbor/morph/label_parse.hpp>

#include "util/strprintf.hpp"

namespace arb {
namespace cv {

parse_error::parse_error(const std::string& msg):
    arbor_exception(msg)
{}

namespace {

struct nil_tag {};

template <typename T>
bool match(const std::type_info& info) {
    return info == typeid(T);
}

template <>
bool match<double>(const std::type_info& info) {
    return info == typeid(double) || info == typeid(int);
}

template <typename T>
T eval_cast(std::any arg) {
    return std::move(std::any_cast<T&>(arg));
}

template <>
double eval_cast<double>(std::any arg) {
    if (arg.type()==typeid(int))    return std::any_cast<int>(arg);
    if (arg.type()==typeid(double)) return std::any_cast<double>(arg);
    throw arb::arbor_internal_error("Bad cast");
}

template <>
cv_policy eval_cast<cv_policy>(std::any arg) {
    if (arg.type()==typeid(cv_policy)) return std::any_cast<cv_policy>(arg);
    throw arb::arbor_internal_error("Bad cast");
}

template <>
region eval_cast<region>(std::any arg) {
    if (arg.type()==typeid(region)) return std::any_cast<region>(arg);
    throw arb::arbor_internal_error("Bad cast");
}

template <>
locset eval_cast<locset>(std::any arg) {
    if (arg.type()==typeid(locset)) return std::any_cast<locset>(arg);
    throw arb::arbor_internal_error("Bad cast");
}

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

template <typename T>
struct fold_eval {
    using fold_fn = std::function<T(T, T)>;
    fold_fn f;

    using anyvec = std::vector<std::any>;
    using iterator = anyvec::iterator;

    fold_eval(fold_fn f): f(std::move(f)) {}

    T fold_impl(iterator left, iterator right) {
        if (std::distance(left,right)==1u) {
            return eval_cast<T>(std::move(*left));
        }
        return f(eval_cast<T>(std::move(*left)), fold_impl(left+1, right));
    }

    std::any operator()(anyvec args) {
        return fold_impl(args.begin(), args.end());
    }
};

template <typename T>
struct fold_match {
    using anyvec = std::vector<std::any>;
    bool operator()(const anyvec& args) const {
        if (args.size()<2u) return false;
        for (auto& a: args) {
            if (!match<T>(a.type())) return false;
        }
        return true;
    }
};

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
};

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

template <typename T>
struct make_fold {
    evaluator state;

    template <typename F>
    make_fold(F&& f, const char* msg="fold"):
        state(fold_eval<T>(std::forward<F>(f)), fold_match<T>(), msg)
    {}

    operator evaluator() const {
        return state;
    }
};

std::unordered_multimap<std::string, evaluator>
eval_map {{"default",
           make_call<>([] () { return cv_policy{default_cv_policy()}; },
                       "'default' with no arguments")},
          {"every-segment",
           make_call<>([] () { return cv_policy{cv_policy_every_segment()}; },
                       "'every-segment' with no arguments")},
          {"every-segment",
           make_call<region>([] (const region& r) { return cv_policy{cv_policy_every_segment(r) }; },
                             "'every-segment' with one argument (every-segment <region-expr>)")},
          {"fixed-per-branch",
           make_call<int>([] (int i) { return cv_policy{cv_policy_fixed_per_branch(i) }; },
                           "'every-segment' with one argument (fixed-per-branch <int>)")},
          {"fixed-per-branch",
           make_call<int, region>([] (int i, const region& r) { return cv_policy{cv_policy_fixed_per_branch(i, r) }; },
                                  "'every-segment' with two arguments (fixed-per-branch <int> <region-expr>)")},
          {"fixed-per-branch",
           make_call<int, region, int>([] (int i, const region& r, int f) { return cv_policy{cv_policy_fixed_per_branch(i, r, f) }; },
                                       "'fixed-per-branch' with three arguments (fixed-per-branch <int> <region-expr> <int>)")},
          {"max-extent",
           make_call<double>([] (double i) { return cv_policy{cv_policy_max_extent(i) }; },
                             "'max-extent' with one argument (max-extent <double>)")},
          {"max-extent",
           make_call<double, region>([] (double i, const region& r) { return cv_policy{cv_policy_max_extent(i, r) }; },
                                     "'max-extent' with two arguments (max-extent <double> <region-expr>)")},
          {"max-extent",
           make_call<double, region, int>([] (double i, const region& r, int f) { return cv_policy{cv_policy_max_extent(i, r, f) }; },
                                          "'max-extent' with three arguments (max-extent <double> <region-expr> <int>)")},
          {"single",
           make_call<>([] () { return cv_policy{cv_policy_single()}; },
                       "'single' with no arguments")},
          {"single",
           make_call<region>([] (const region& r) { return cv_policy{cv_policy_single(r) }; },
                             "'single' with one argument (single <region-expr>)")},
          {"explicit",
           make_call<locset>([] (const locset& l) { return cv_policy{cv_policy_explicit(l) }; },
                             "'explicit' with one argument (explicit <locset-expr>)")},
          {"explicit",
           make_call<locset, region>([] (const locset& l, const region& r) { return cv_policy{cv_policy_explicit(l, r) }; },
                                     "'explicit' with one argument (explicit <locset-expr> <region-expr>)")},
          {"join",
           make_fold<cv_policy>([](cv_policy l, cv_policy r) { return l + r; },
                                "'add' with at least 2 arguments: (join cv_policy cv_policy ...)")},
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
    std::string msg =
        util::pprintf("'{}' with {} argument{}",
                      name, nargs,
                      nargs==0?"s": nargs==1u?":": "s:");
    if (nargs) {
        msg += " (";
        bool first = true;
        for (auto& a: args) {
            msg += util::pprintf("{}{}", first?"":" ", type_string(a.type()));
            first = false;
        }
        msg += ")";
    }
    return msg;
}

parse_error make_parse_error(std::string const& msg, src_location loc) {
    return {util::pprintf("error in CV policy description at {}: {}.", loc, msg)};
}

// Evaluate an s expression.
// On success the result is wrapped in std::any, where the result is one of:
//      int:            an integer atom
//      double:         a real atom
//      cv_policy: a region
//
// If there invalid input is detected, hopefully return value contains
// a cv_policy_error_state with an error string and location.
//
// If there was an unexpected/fatal error, an exception will be thrown.
parse_hopefully<std::any> eval(const s_expr& e) {
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
                return {std::string(t.spelling)};
            // An arbitrary symbol in a CV policy expression is an error.
            case tok::symbol:
                return util::unexpected(make_parse_error(
                        util::pprintf("Unexpected symbol '{}' in a CV policy definition.", e),
                        location(e)));
            case tok::error:
                return util::unexpected(make_parse_error(e.atom().spelling, location(e)));
            default:
                return util::unexpected(make_parse_error(util::pprintf("Unexpected term '{}' in a CV policy definition", e), location(e)));
        }
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

        // if no matches found, maybe this is a locset?
        if (matches.first == matches.second) {
            auto lbl = parse_label_expression(e);
            if (lbl.has_value()) {
                return { lbl.value() };
            } else {
                return util::unexpected(make_parse_error(lbl.error().what(), location(e)));
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
            auto msg = util::pprintf("No matches for {}", eval_description(name.c_str(), *args));
            msg += util::pprintf("\n  There are {} potential candiate{}", nc, nc?":":".");
            int count = 0;
            for (auto i=matches.first; i!=matches.second; ++i) {
                msg += util::pprintf("\n  Candidate {}: {}", ++count, i->second.message);
            }

            return util::unexpected(make_parse_error(msg, location(e)));
        }
    }

    return util::unexpected(make_parse_error(
            util::pprintf("'{}' is not either integer, real expression of the form (op <args>)", e),
            location(e)));
}
}

parse_hopefully<cv_policy> parse_expression(const std::string& s) {
    if (auto e = eval(parse_s_expr(s))) {
        if (e->type() == typeid(cv_policy)) {
            return {std::move(std::any_cast<cv_policy&>(*e))};
        }
        return util::unexpected(
                parse_error(
                util::pprintf("Invalid description: '{}' is not a valid CV policy expression.", s)));
    }
    else {
        return util::unexpected(parse_error(std::string()+e.error().what()));
    }
}

} // namespace cv
} // namespace arb

