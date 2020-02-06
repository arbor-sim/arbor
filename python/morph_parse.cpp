#include <arbor/util/any.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/morph/locset.hpp>

#include "error.hpp"
#include "s_expr.hpp"
#include "morph_parse.hpp"

namespace pyarb {

struct nil_tag {};

template <typename T>
bool match(const std::type_info& info) {
    return info == typeid(T);
}

template <>
bool match<double>(const std::type_info& info) {
    return info == typeid(double) || info == typeid(int);
}

template <>
bool match<arb::region>(const std::type_info& info) {
    return info == typeid(arb::region) || info == typeid(nil_tag);
}

template <>
bool match<arb::locset>(const std::type_info& info) {
    return info == typeid(arb::locset) || info == typeid(nil_tag);
}

template <typename T>
T eval_cast(arb::util::any arg) {
    return std::move(arb::util::any_cast<T&>(arg));
}

template <>
double eval_cast<double>(arb::util::any arg) {
    if (arg.type()==typeid(int)) return arb::util::any_cast<int>(arg);
    return arb::util::any_cast<double>(arg);
}

template <>
arb::region eval_cast<arb::region>(arb::util::any arg) {
    if (arg.type()==typeid(arb::region)) return arb::util::any_cast<arb::region>(arg);
    return arb::reg::nil();
}

template <>
arb::locset eval_cast<arb::locset>(arb::util::any arg) {
    if (arg.type()==typeid(arb::locset)) return arb::util::any_cast<arb::locset>(arg);
    return arb::ls::nil();
}

template <typename... Args>
struct call_eval {
    using ftype = std::function<arb::util::any(Args...)>;
    ftype f;
    call_eval(ftype f): f(std::move(f)) {}

    template<std::size_t... I>
    arb::util::any expand_args_then_eval(std::vector<arb::util::any> args, std::index_sequence<I...>) {
        return f(eval_cast<Args>(std::move(args[I]))...);
    }

    arb::util::any operator()(std::vector<arb::util::any> args) {
        return expand_args_then_eval(std::move(args), std::make_index_sequence<sizeof...(Args)>());
    }
};

template <typename... Args>
struct call_match {
    template <std::size_t I, typename T, typename Q, typename... Rest>
    bool match_args_impl(const std::vector<arb::util::any>& args) const {
        return match<T>(args[I].type()) && match_args_impl<I+1, Q, Rest...>(args);
    }

    template <std::size_t I, typename T>
    bool match_args_impl(const std::vector<arb::util::any>& args) const {
        return match<T>(args[I].type());
    }

    template <std::size_t I>
    bool match_args_impl(const std::vector<arb::util::any>& args) const {
        return true;
    }

    bool operator()(const std::vector<arb::util::any>& args) const {
        const auto nargs_in = args.size();
        const auto nargs_ex = sizeof...(Args);
        return nargs_in==nargs_ex? match_args_impl<0, Args...>(args): false;
    }
};

template <typename T>
struct fold_eval {
    using fold_fn = std::function<T(T, T)>;
    fold_fn f;

    using anyvec = std::vector<arb::util::any>;
    using iterator = anyvec::iterator;

    fold_eval(fold_fn f): f(std::move(f)) {}

    T fold_impl(iterator left, iterator right) {
        if (std::distance(left,right)==1u) {
            return eval_cast<T>(std::move(*left));
        }
        return f(eval_cast<T>(std::move(*left)), fold_impl(left+1, right));
    }

    arb::util::any operator()(anyvec args) {
        return fold_impl(args.begin(), args.end());
    }
};

template <typename T>
struct fold_match {
    using anyvec = std::vector<arb::util::any>;
    bool operator()(const anyvec& args) const {
        if (args.size()<2u) return false;
        for (auto& a: args) {
            if (!match<T>(a.type())) return false;
        }
        return true;
    }
};

struct evaluator {
    using any_vec = std::vector<arb::util::any>;
    using eval_fn = std::function<arb::util::any(any_vec)>;
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

std::unordered_multimap<std::string, evaluator> eval_map {
    // Functions that return regions
    {"nil",     make_call<>(arb::reg::nil,
                            "'nil' with 0 arguments")},
    {"all",     make_call<>(arb::reg::all,
                            "'all' with 0 arguments")},
    {"tag",     make_call<int>(arb::reg::tagged,
                            "'tag' with 1 argment: (tag_id:integer)")},
    {"branch",  make_call<int>(arb::reg::branch,
                            "'branch' with 1 argument: (branch_id:integer)")},
    {"cable",   make_call<int, double, double>(arb::reg::cable,
                            "'cable' with 3 arguments: (branch_id:integer prox:real dist:real)")},
    {"region",  make_call<std::string>(arb::reg::named,
                            "'region' with 1 argument: (name:string)")},
    {"distal_interval",  make_call<arb::locset, double>(arb::reg::distal_interval,
                            "'distal_interval' with 2 arguments: (start:locset extent:real)")},
    {"proximal_interval",make_call<arb::locset, double>(arb::reg::proximal_interval,
                            "'proximal_interval' with 2 arguments: (start:locset extent:real)")},
    {"radius_lt",make_call<arb::region, double>(arb::reg::radius_lt,
                            "'radius_lt' with 2 arguments: (reg:region radius:real)")},
    {"radius_le",make_call<arb::region, double>(arb::reg::radius_le,
                            "'radius_le' with 2 arguments: (reg:region radius:real)")},
    {"radius_gt",make_call<arb::region, double>(arb::reg::radius_gt,
                            "'radius_gt' with 2 arguments: (reg:region radius:real)")},
    {"radius_ge",make_call<arb::region, double>(arb::reg::radius_ge,
                            "'radius_ge' with 2 arguments: (reg:region radius:real)")},
    {"z_dist_from_root_lt",make_call<double>(arb::reg::z_dist_from_root_lt,
                            "'z_dist_from_root_lt' with 1 arguments: (distance:real)")},
    {"z_dist_from_root_le",make_call<double>(arb::reg::z_dist_from_root_le,
                            "'z_dist_from_root_le' with 1 arguments: (distance:real)")},
    {"z_dist_from_root_gt",make_call<double>(arb::reg::z_dist_from_root_gt,
                            "'z_dist_from_root_gt' with 1 arguments: (distance:real)")},
    {"z_dist_from_root_ge",make_call<double>(arb::reg::z_dist_from_root_ge,
                            "'z_dist_from_root_ge' with 1 arguments: (distance:real)")},
    {"join",    make_fold<arb::region>(static_cast<arb::region(*)(arb::region, arb::region)>(arb::join),
                            "'join' with at least 2 arguments: (region region [...region])")},
    {"intersect",make_fold<arb::region>(static_cast<arb::region(*)(arb::region, arb::region)>(arb::intersect),
                            "'intersect' with at least 2 arguments: (region region [...region])")},

    // Functions that return locsets
    {"root",    make_call<>(arb::ls::root,
                            "'root' with 0 arguments")},

    {"location", make_call<int, double>([](int bid, double pos){return arb::ls::location(arb::msize_t(bid), pos);},
                            "'location' with 2 arguments: (branch_id:integer position:real)")},
    {"terminal", make_call<>(arb::ls::terminal,
                            "'terminal' with 0 arguments")},
    {"sample",  make_call<int>(arb::ls::sample,
                            "'sample' with 1 argument: (sample_id:integer)")},
    {"distal",  make_call<arb::region>(arb::ls::most_distal,
                            "'distal' with 1 argument: (reg:region)")},
    {"proximal",make_call<arb::region>(arb::ls::most_proximal,
                            "'proximal' with 1 argument: (reg:region)")},
    {"uniform",make_call<arb::region, int, int, int>(arb::ls::uniform,
                            "'uniform' with 4 arguments: (reg:region, first:int, last:int, seed:int)")},
    {"on_branches",make_call<double>(arb::ls::on_branches,
                            "'on_branches' with 1 argument: (pos:double)")},
    {"locset",  make_call<std::string>(arb::ls::named,
                            "'locset' with 1 argument: (name:string)")},
    {"join",    make_fold<arb::locset>(static_cast<arb::locset(*)(arb::locset, arb::locset)>(arb::join),
                            "'join' with at least 2 arguments: (locset locset [...locset])")},
    {"sum",     make_fold<arb::locset>(static_cast<arb::locset(*)(arb::locset, arb::locset)>(arb::sum),
                            "'sum' with at least 2 arguments: (locset locset [...locset])")},
};

parse_hopefully<arb::util::any> eval(const s_expr& e);

parse_hopefully<std::vector<arb::util::any>> eval_args(const s_expr& e) {
    if (!e) return {std::vector<arb::util::any>{}}; // empty argument list
    const s_expr* h = &e;
    std::vector<arb::util::any> args;
    while (*h) {
        auto arg = eval(h->head());
        if (!arg) return std::move(arg.error());
        args.push_back(std::move(*arg));
        h = &h->tail();
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
std::string eval_description(const char* name, const std::vector<arb::util::any>& args) {
    auto type_string = [](const std::type_info& t) -> const char* {
        if (t==typeid(int))         return "integer";
        if (t==typeid(double))      return "real";
        if (t==typeid(arb::region)) return "region";
        if (t==typeid(arb::locset)) return "locset";
        if (t==typeid(nil_tag))     return "()";
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

// Evaluate an s expression.
// On success the result is wrapped in util::any, where the result is one of:
//      int         : an integer atom
//      double      : a real atom
//      arb::region : a region
//      arb::locset : a locset
//
// If there invalid input is detected, hopefully return value contains
// a parse_error_state with an error string and location.
//
// If there was an unexpected/fatal error, an exception will be thrown.
parse_hopefully<arb::util::any> eval(const s_expr& e) {
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
            case tok::error:
                return parse_error_state{e.atom().spelling, location(e)};
            default:
                return parse_error_state{
                    util::pprintf("Unexpected term: {}", e), location(e)};
        }
    }
    if (e.head().is_atom()) {
        // This must be a function evaluation, where head is the function name, and
        // tail is a list of arguments.

        // Evaluate the arguments, and return error state if an error ocurred.
        auto args = eval_args(e.tail());
        if (!args) {
            return args.error();
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
        auto msg = util::pprintf("No matches for {}", eval_description(name.c_str(), *args));
        msg += util::pprintf("\n  There are {} potential candiates{}", nc, nc?":":".");
        int count = 0;
        for (auto i=matches.first; i!=matches.second; ++i) {
            msg += util::pprintf("\n  Candidate {}  {}", ++count, i->second.message);
        }
        return parse_error_state{std::move(msg), location(e)};
    }

    return parse_error_state{
        util::pprintf("Unable to evaluate '{}': expression must be either integer, real expression of the form (op <args>)", e),
            location(e)};
}

} // namespace pyarb


