#pragma once

#include <any>
#include <string>
#include <sstream>
#include <iostream>
#include <typeinfo>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/util/expected.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/iexpr.hpp>

namespace arborio {
using namespace arb;

// Check typeinfo against expected types
template <typename T>
bool match(const std::type_info& info) { return info == typeid(T); }
template <> inline
bool match<double>(const std::type_info& info) { return info == typeid(double) || info == typeid(int); }
template <> inline
bool match<arb::locset>(const std::type_info& info) { return info == typeid(arb::locset); }
template <> inline
bool match<arb::region>(const std::type_info& info) { return info == typeid(arb::region); }
template <> inline
bool match<arb::iexpr>(const std::type_info& info) { return info == typeid(arb::iexpr); }

// Convert a value wrapped in a std::any to target type.
template <typename T>
T eval_cast(std::any arg) {
    return std::move(std::any_cast<T&>(arg));
}

template <> inline
double eval_cast<double>(std::any arg) {
    if (arg.type()==typeid(int)) return std::any_cast<int>(arg);
    return std::any_cast<double>(arg);
}

template <typename T>
T conversion_cast(std::any arg) {
    return eval_cast<T>(std::move(arg));
}

template <typename T, typename Q, typename... Types>
T conversion_cast(std::any arg) {
    if (match<Q>(arg.type())) return T(eval_cast<Q>(arg));

    return conversion_cast<T, Types...>(arg);
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


// Fold with first converting from any of the "Types" to type T.
template <typename T, typename... Types>
struct fold_conversion_eval {
    using fold_fn = std::function<T(T, T)>;
    fold_fn f;

    using anyvec = std::vector<std::any>;
    using iterator = anyvec::iterator;

    fold_conversion_eval(fold_fn f): f(std::move(f)) {}

    T fold_impl(iterator left, iterator right) {
        if (std::distance(left,right)==1u) {
            return conversion_cast<T, Types...>(std::move(*left));
        }
        // Compute fold. Order is important for left-associative operations like division and
        // subtraction
        auto back = right - 1;
        return f(fold_impl(left, back), conversion_cast<T, Types...>(std::move(*back)));
    }

    std::any operator()(anyvec args) {
        return fold_impl(args.begin(), args.end());
    }
};

// Test if all args match at least one of the "Types" types
template <typename... Types>
struct fold_conversion_match {
    bool operator()(const std::vector<std::any>& args) const {
        if (args.size() < 2) return false;

        bool m = true;
        for (const auto& a : args) {
            m = m && (match<Types>(a.type()) || ...);
        }
        return m;
    }
};

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

// Fold with first converting from any of the "Types" to type T.
template <typename T, typename... Types>
struct make_conversion_fold {
    evaluator state;

    template <typename F>
    make_conversion_fold(F&& f, const char* msg = "fold_conversion"):
        state(fold_conversion_eval<T, Types...>(std::forward<F>(f)),
            fold_conversion_match<T, Types...>(),
            msg) {}

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

// Convert a value wrapped in a std::any to an optional std::variant type
template <typename T, std::size_t I=0>
std::optional<T> eval_cast_variant(const std::any& a) {
    if constexpr (I<std::variant_size_v<T>) {
        using var_type = std::variant_alternative_t<I, T>;
        return match<var_type>(a.type())? eval_cast<var_type>(a): eval_cast_variant<T, I+1>(a);
    }
    return std::nullopt;
}

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

template<typename... Ts>
std::string concat(Ts... ts) {
    std::stringstream ss;
    (ss << ... << ts);
    return ss.str();
}

// try to parse an atom
template<typename E>
util::expected<std::any, E> eval_atom(const s_expr& e) {
    arb_assert(e.is_atom());
    auto& t = e.atom();
    switch (t.kind) {
        case tok::integer:
            return {std::stoi(t.spelling)};
        case tok::real:
            return {std::stod(t.spelling)};
        case tok::string:
            return {std::string(t.spelling)};
        case tok::symbol:
            return util::unexpected(E(concat("Unexpected symbol '", e, "' in definition."), location(e)));
        case tok::error:
            return util::unexpected(E(e.atom().spelling, location(e)));
        default:
            return util::unexpected(E(concat("Unexpected term '", e, "' in definition"), location(e)));
    }
}
}
