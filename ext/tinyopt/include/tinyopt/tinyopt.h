#pragma once

#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

#define TINYOPT_VERSION "1.0"
#define TINYOPT_VERSION_MAJOR 1
#define TINYOPT_VERSION_MINOR 0
#define TINYOPT_VERSION_PATCH 0
#define TINYOPT_VERSION_PRERELEASE ""

namespace to {

// maybe<T> represents an optional value of type T,
// with an interface similar to C++17 std::optional.
//
// Other than C++14 compatibility, the main deviations/extensions
// from std::optional are:
//
// 1. maybe<void> represents an optional value of an 'empty' type.
//    This is used primarily for consistency in generic uses of
//    maybe in contexts where functions may not return or consume
//    a value.
//
// 2. Monadic-like overloads of operator<< that lift a function
//    or a functional object on the left hand side when the
//    right hand side is maybe<T> (see below). With an lvalue
//    on the left hand side, operator<< acts as a conditional
//    assignment.
//
// nothing is a special value that converts to an empty maybe<T> for any T.

constexpr struct nothing_t {} nothing;

template <typename T>
struct maybe {
    maybe() noexcept: ok(false) {}
    maybe(nothing_t) noexcept: ok(false) {}
    maybe(const T& v): ok(true) { construct(v); }
    maybe(T&& v): ok(true) { construct(std::move(v)); }
    maybe(const maybe& m): ok(m.ok) { if (ok) construct(*m); }
    maybe(maybe&& m): ok(m.ok) { if (ok) construct(std::move(*m)); }

    ~maybe() { destroy(); }

    template <typename U>
    maybe(const maybe<U>& m): ok(m.ok) { if (ok) construct(*m); }

    template <typename U>
    maybe(maybe<U>&& m): ok(m.ok) { if (ok) construct(std::move(*m)); }

    maybe& operator=(nothing_t) { return destroy(), *this; }
    maybe& operator=(const T& v) { return assign(v), *this; }
    maybe& operator=(T&& v) { return assign(std::move(v)), *this; }
    maybe& operator=(const maybe& m) { return m.ok? assign(*m): destroy(), *this; }
    maybe& operator=(maybe&& m) { return m.ok? assign(std::move(*m)): destroy(), *this; }

    const T& value() const & { return assert_ok(), *vptr(); }
    T&& value() && { return assert_ok(), std::move(*vptr()); }

    const T& operator*() const & noexcept { return *vptr(); }
    const T* operator->() const & noexcept { return vptr(); }
    T&& operator*() && { return std::move(*vptr()); }

    bool has_value() const noexcept { return ok; }
    explicit operator bool() const noexcept { return ok; }

    template <typename> friend struct maybe;

private:
    bool ok = false;
    alignas(T) char data[sizeof(T)];

    T* vptr() noexcept { return reinterpret_cast<T*>(data); }
    const T* vptr() const noexcept { return reinterpret_cast<const T*>(data); }

    void construct(const T& v) { new (data) T(v); ok = true; }
    void construct(T&& v) { new (data) T(std::move(v)); ok = true; }

    void assign(const T& v) { if (ok) *vptr()=v; else construct(v); }
    void assign(T&& v) { if (ok) *vptr()=std::move(v); else construct(std::move(v)); }

    void destroy() { if (ok) (**this).~T(); ok = false; }
    void assert_ok() const { if (!ok) throw std::invalid_argument("is nothing"); }
};

namespace impl {
    template <typename T>
    struct is_maybe_: std::false_type {};

    template <typename T>
    struct is_maybe_<maybe<T>>: std::true_type {};
}

template <typename T>
using is_maybe = impl::is_maybe_<std::remove_cv_t<std::remove_reference_t<T>>>;

// maybe<void> acts as a maybe<T> with an empty or inaccessible wrapped value.

template <>
struct maybe<void> {
    bool ok = false;

    constexpr maybe() noexcept: ok(false) {}
    constexpr maybe(nothing_t&) noexcept: ok(false) {}
    constexpr maybe(const nothing_t&) noexcept: ok(false) {}
    constexpr maybe(nothing_t&&) noexcept: ok(false) {}
    constexpr maybe(const maybe<void>& m) noexcept: ok(m.ok) {}

    template <typename X, typename = std::enable_if_t<!is_maybe<X>::value>>
    constexpr maybe(X&&) noexcept: ok(true) {}

    template <typename U>
    constexpr maybe(const maybe<U>& m) noexcept: ok(m.ok) {}

    maybe& operator=(nothing_t) noexcept { return ok = false, *this; }
    maybe& operator=(const maybe& m) noexcept { return ok = m.ok, *this; }
    template <typename U>
    maybe& operator=(U&&) noexcept { return ok = true, *this; }

    bool has_value() const noexcept { return ok; }
    constexpr explicit operator bool() const noexcept { return ok; }
};

// something is a non-empty maybe<void> value.

constexpr maybe<void> something(true);

// just<X> converts a value of type X to a maybe<X> containing the value.

template <typename X>
auto just(X&& x) { return maybe<std::decay_t<X>>(std::forward<X>(x)); }

// operator<< offers monadic-style chaining of maybe<X> values:
// (f << m) evaluates to an empty maybe if m is empty, or else to
// a maybe<V> value wrapping the result of applying f to the value
// in m.

template <
    typename F,
    typename T,
    typename R = std::decay_t<decltype(std::declval<F>()(std::declval<const T&>()))>,
    typename = std::enable_if_t<std::is_same<R, void>::value>
>
maybe<void> operator<<(F&& f, const maybe<T>& m) {
    if (m) return f(*m), something; else return nothing;
}

template <
    typename F,
    typename T,
    typename R = std::decay_t<decltype(std::declval<F>()(std::declval<const T&>()))>,
    typename = std::enable_if_t<!std::is_same<R, void>::value>
>
maybe<R> operator<<(F&& f, const maybe<T>& m) {
    if (m) return f(*m); else return nothing;
}

template <
    typename F,
    typename R = std::decay_t<decltype(std::declval<F>()())>,
    typename = std::enable_if_t<std::is_same<R, void>::value>
>
maybe<void> operator<<(F&& f, const maybe<void>& m) {
    return m? (f(), something): nothing;
}

template <
    typename F,
    typename R = std::decay_t<decltype(std::declval<F>()())>,
    typename = std::enable_if_t<!std::is_same<R, void>::value>
>
maybe<R> operator<<(F&& f, const maybe<void>& m) {
    return m? just(f()): nothing;
}

// If the lhs is not functional, return a maybe value with the result
// of assigning the value in the rhs, or nothing if the rhs is nothing.

template <typename T, typename U>
auto operator<<(T& x, const maybe<U>& m) -> maybe<std::decay_t<decltype(x=*m)>> {
    if (m) return x=*m; else return nothing;
}

template <typename T>
auto operator<<(T& x, const maybe<void>& m) -> maybe<std::decay_t<decltype(x=true)>> {
    if (m) return x=true; else return nothing;
}

// Tinyopt exceptions, usage, error reporting functions:

// `option_error` is the base class for exceptions thrown
// by the option handling functions.

struct option_error: public std::runtime_error {
    option_error(const std::string& message): std::runtime_error(message) {}
    option_error(const std::string& message, std::string arg):
        std::runtime_error(message+": "+arg), arg(std::move(arg)) {}

    std::string arg;
};

struct option_parse_error: option_error {
    option_parse_error(const std::string &arg):
        option_error("option parse error", arg) {}
};

struct missing_mandatory_option: option_error {
    missing_mandatory_option(const std::string &arg):
        option_error("missing mandatory option", arg) {}
};

struct missing_argument: option_error {
    missing_argument(const std::string &arg):
        option_error("option misssing argument", arg) {}
};

struct user_option_error: option_error {
    user_option_error(const std::string &arg):
        option_error(arg) {}
};

// `usage` prints usage information to stdout (no error message)
// or to stderr (with error message). It extracts the program basename
// from the provided argv[0] string.

inline void usage(const char* argv0, const std::string& usage_str, const std::string& prefix = "Usage: ") {
    const char* basename = std::strrchr(argv0, '/');
    basename = basename? basename+1: argv0;

    std::cout << prefix << basename << " " << usage_str << "\n";
}

inline void usage_error(const char* argv0, const std::string& usage_str, const std::string& parse_err, const std::string& prefix = "Usage: ") {
    const char* basename = std::strrchr(argv0, '/');
    basename = basename? basename+1: argv0;

    std::cerr << basename << ": " << parse_err << "\n";
    std::cerr << prefix << basename << " " << usage_str << "\n";
}

// Parser objects act as functionals, taking
// a const char* argument and returning maybe<T>
// for some T.

template <typename V>
struct default_parser {
    maybe<V> operator()(const char* text) const {
        if (!text) return nothing;
        V v;
        std::istringstream stream(text);
        if (!(stream >> v)) return nothing;
	if (!stream.eof()) stream >> std::ws;
	return stream.eof()? maybe<V>(v): nothing;
    }
};

template <>
struct default_parser<const char*> {
    maybe<const char*> operator()(const char* text) const {
        return just(text);
    }
};

template <>
struct default_parser<std::string> {
    maybe<std::string> operator()(const char* text) const {
        return just(std::string(text));
    }
};

template <>
struct default_parser<void> {
    maybe<void> operator()(const char*) const {
        return something;
    }
};

template <typename V>
class keyword_parser {
    std::vector<std::pair<std::string, V>> map_;

public:
    template <typename KeywordPairs>
    keyword_parser(const KeywordPairs& pairs) {
        using std::begin;
        using std::end;
        map_.assign(begin(pairs), end(pairs));
    }

    maybe<V> operator()(const char* text) const {
        if (!text) return nothing;
        for (const auto& p: map_) {
            if (text==p.first) return p.second;
        }
        return nothing;
    }
};

// Returns a parser that matches a set of keywords,
// returning the corresponding values in the supplied
// pairs.

template <typename KeywordPairs>
auto keywords(const KeywordPairs& pairs) {
    using std::begin;
    using value_type = std::decay_t<decltype(std::get<1>(*begin(pairs)))>;
    return keyword_parser<value_type>(pairs);
}


// A parser for delimited sequences of values; returns
// a vector of the values obtained from the supplied
// per-item parser.

template <typename P>
class delimited_parser {
    char delim_;
    P parse_;
    using inner_value_type = std::decay_t<decltype(*std::declval<P>()(""))>;

public:
    template <typename Q>
    delimited_parser(char delim, Q&& parse): delim_(delim), parse_(std::forward<Q>(parse)) {}

    maybe<std::vector<inner_value_type>> operator()(const char* text) const {
        if (!text) return nothing;

        std::vector<inner_value_type> values;
        if (!*text) return values;

        std::size_t n = std::strlen(text);
        std::vector<char> input(1+n);
        std::copy(text, text+n, input.data());

        char* p = input.data();
        char* end = input.data()+1+n;
        do {
            char* q = p;
            while (*q && *q!=delim_) ++q;
            *q++ = 0;

            if (auto mv = parse_(p)) values.push_back(*mv);
            else return nothing;

            p = q;
        } while (p<end);

        return values;
    }
};

// Convenience constructors for delimited parser.

template <typename Q>
auto delimited(char delim, Q&& parse) {
    using P = std::decay_t<Q>;
    return delimited_parser<P>(delim, std::forward<Q>(parse));
}

template <typename V>
auto delimited(char delim = ',') {
    return delimited(delim, default_parser<V>{});
}

// Option keys
// -----------
//
// A key is how the option is specified in an argument list, and is typically
// represented as a 'short' (e.g. '-a') option or a 'long' option (e.g.
// '--apple').
//
// The value for an option can always be taken from the next argument in the
// list, but in addition can be specified together with the key itself,
// depending on the properties of the option key:
//
//     --key=value         'Long' style argument for key "--key"
//     -kvalue             'Compact' style argument for key "-k"
//
// Compact option keys can be combined in the one item in the argument list, if
// the options do not take any values (that is, they are flags). For example,
// if -a, -b are flags and -c takes an integer argument, with all three keys
// marked as compact, then an item '-abc3' in the argument list will be parsed
// in the same way as the sequence of items '-a', '-b', '-c', '3'.
//
// An option without a key will match any item in the argument list; options
// with keys are always checked first.
//
// Only short and long kets can be used with to::parse.

struct key {
    std::string label;
    enum style { shortfmt, longfmt, compact } style = shortfmt;

    key(std::string l): label(std::move(l)) {
        if (label[0]=='-' && label[1]=='-') style = longfmt;
    }

    key(const char* label): key(std::string(label)) {}

    key(std::string label, enum style style):
        label(std::move(label)), style(style) {}
};

inline namespace literals {

inline key operator""_short(const char* label, std::size_t) {
    return key(label, key::shortfmt);
}

inline key operator""_long(const char* label, std::size_t) {
    return key(label, key::longfmt);
}

inline key operator""_compact(const char* label, std::size_t) {
    return key(label, key::compact);
}

} // namespace literals

// Argument state
// --------------
//
// to::state represents the collection of command line arguments. Mutating
// operations (shift(), successful option matching, etc.) will modify the
// underlying set of arguments used to construct the state object.
//
// This is used only internally — it is not part of the public API.
// Members are left public for the purpose of unit testing.

struct state {
    int& argc;
    char** argv;
    unsigned optoff = 0;

    state(int& argc, char** argv): argc(argc), argv(argv) {}

    // False => no more arguments.
    explicit operator bool() const { return *argv; }

    // Shift arguments left in-place.
    void shift(unsigned n = 1) {
        char** skip = argv;
        while (*skip && n) ++skip, --n;

        argc -= (skip-argv);
        auto p = argv;
        do { *p++ = *skip; } while (*skip++);

        optoff = 0;
    }

    // Skip current argument without modifying list.
    void skip() {
         if (*argv) ++argv;
    }

    // Match an option given by the key which takes an argument.
    // If successful, consume option and argument and return pointer to
    // argument string, else return nothing.
    maybe<const char*> match_option(const key& k) {
        const char* p = nullptr;

        if (k.style==key::compact) {
            if ((p = match_compact_key(k.label.c_str()))) {
                if (!*p) {
                    p = argv[1];
                    shift(2);
                }
                else shift();
                return p;
            }
        }
        else if (!optoff && k.label==*argv) {
            p = argv[1];
            shift(2);
            return p;
        }
        else if (!optoff && k.style==key::longfmt) {
            auto keylen = k.label.length();
            if (!std::strncmp(*argv, k.label.c_str(), keylen) && (*argv)[keylen]=='=') {
                p = &(*argv)[keylen+1];
                shift();
                return p;
            }
        }

        return nothing;
    }

    // Match a flag given by the key.
    // If successful, consume flag and return true, else return false.
    bool match_flag(const key& k) {
        if (k.style==key::compact) {
            if (auto p = match_compact_key(k.label.c_str())) {
                if (!*p) shift();
                return true;
            }
        }
        else if (!optoff && k.label==*argv) {
            shift();
            return true;
        }

        return false;
    }

    // Compact-style keys can be combined in one argument; combined keys
    // with a common prefix only need to supply the prefix once at the
    // beginning of the argument.
    const char* match_compact_key(const char* k) {
        unsigned keylen = std::strlen(k);

        unsigned prefix_max = std::min(keylen-1, optoff);
        for (std::size_t l = 0; l<=prefix_max; ++l) {
            if (l && strncmp(*argv, k, l)) break;
            if (strncmp(*argv+optoff, k+l, keylen-l)) continue;
            optoff += keylen-l;
            return *argv+optoff;
        }

        return nullptr;
    }
};

// Sinks and actions
// -----------------
//
// Sinks wrap a function that takes a pointer to an option parameter and stores
// or acts upon the parsed result.
//
// They can be constructed from an lvalue reference or a functional object (via
// the `action` function) with or without an explicit parser function. If no
// parser is given, a default one is used if the correct value type can be
// determined.

namespace impl {
    template <typename T> struct fn_arg_type { using type = void; };
    template <typename R, typename X> struct fn_arg_type<R (X)> { using type = X; };
    template <typename R, typename X> struct fn_arg_type<R (*)(X)> { using type = X; };
    template <typename R, typename C, typename X> struct fn_arg_type<R (C::*)(X)> { using type = X; };
    template <typename R, typename C, typename X> struct fn_arg_type<R (C::*)(X) const> { using type = X; };
    template <typename R, typename C, typename X> struct fn_arg_type<R (C::*)(X) volatile> { using type = X; };
    template <typename R, typename C, typename X> struct fn_arg_type<R (C::*)(X) const volatile> { using type = X; };

    template <typename...> struct void_type { using type = void; };
}

template <typename T, typename = void>
struct unary_argument_type { using type = typename impl::fn_arg_type<T>::type; };

template <typename T>
struct unary_argument_type<T, typename impl::void_type<decltype(&T::operator())>::type> {
    using type = typename impl::fn_arg_type<decltype(&T::operator())>::type;
};

template <typename T>
using unary_argument_type_t = typename unary_argument_type<T>::type;

struct sink {
    // Tag class for constructor.
    static struct action_t {} action;

    sink():
        sink(action, [](const char*) { return true; })
    {}

    template <typename V>
    sink(V& var): sink(var, default_parser<V>{}) {}

    template <typename V, typename P>
    sink(V& var, P parser):
        sink(action, [ref=std::ref(var), parser](const char* param) {
                if (auto p = parser(param)) return ref.get() = std::move(*p), true;
                else return false;
            })
    {}

    template <typename Action>
    sink(action_t, Action a): op(std::move(a)) {}

    bool operator()(const char* param) const { return op(param); }
    std::function<bool (const char*)> op;

};

// Convenience functions for construction of sink actions
// with explicit or implicit parser.

template <typename F, typename A = std::decay_t<unary_argument_type_t<F>>>
sink action(F f) {
    return sink(sink::action,
        [f = std::move(f)](const char* arg) -> bool {
            return static_cast<bool>(f << default_parser<A>{}(arg));
        });
}

template <typename F, typename P>
sink action(F f, P parser) {
    return sink(sink::action,
        [f = std::move(f), parser = std::move(parser)](const char* arg) -> bool {
            return static_cast<bool>(f << parser(arg));
        });
}

// Special actions:
//
// error(message)    Throw a user_option_error with the supplied message.

inline sink error(std::string message) {
    return sink(sink::action,
        [m = std::move(message)](const char*) -> bool {
            throw user_option_error(m);
        });
}

// Sink adaptors:
//
// These adaptors constitute short cuts for making actions that count the
// occurance of a flag, set a fixed value when a flag is provided, or for
// appending an option parameter onto a vector of values.

// Push parsed option parameter on to container.
template <typename Container, typename P = default_parser<typename Container::value_type>>
sink push_back(Container& c, P parser = P{}) {
    return action(
        [ref = std::ref(c)](typename Container::value_type v) { ref.get().push_back(std::move(v)); },
        std::move(parser));
}

// Set v to value when option parsed; ignore any option parameter.
template <typename V, typename X>
sink set(V& v, X value) {
    return action([ref = std::ref(v), value = std::move(value)] { ref.get() = value; });
}

// Set v to true when option parsed; ignore any option parameter.
template <typename V>
sink set(V& v) {
    return set(v, true);
}

// Incrememnt v when option parsed; ignore any option parameter.
template <typename V>
sink increment(V& v) {
    return action([ref = std::ref(v)] { ++ref.get(); });
}

template <typename V, typename X>
sink increment(V& v, X delta) {
    return action([ref = std::ref(v), delta = std::move(delta)] { ref.get() += delta; });
}

// Modal configuration
// -------------------
//
// Options can be filtered by some predicate, and can trigger a state
// change when successfully processed.
//
// Filter construction:
//     to::when(Fn f)       f is a functional with signature bool (int)
//     to::when(int s)      equivalent to to::when([](int k) { return k==s; })
//     to::when(a, b, ...)  filter than is satisfied by to::when(a) or
//                          to::when(b) or ...
//
// The argument to the filter predicate is the 'mode', a mutable state maintained
// during a single run of to::run().
//
// Mode changes:
//     to::then(Fn f)       f is a functional with signature int (int)
//     to::then(int s)      equivalent to to::then([](int) { return s; })
//
// The argument to the functional is the current mode; the return value
// sets the new value of mode.
//
// Filters are called before keys are matched; modal changes are called
// after an option is processed. All 

using filter = std::function<bool (int)>;
using modal = std::function<int (int)>;

template <typename F, typename = decltype(std::declval<F>()(0))>
filter when(F f) {
    return [f = std::move(f)](int mode) { return static_cast<bool>(f(mode)); };
}

inline filter when(int m) {
    return [m](int mode) { return m==mode; };
}

template <typename A, typename B, typename... Rest>
filter when(A a, B&& b, Rest&&... rest) {
    return
        [fhead = when(std::forward<A>(a)),
         ftail = when(std::forward<B>(b), std::forward<Rest>(rest)...)](int m) {
        return fhead(m) || ftail(m);
    };
}

template <typename F, typename = decltype(std::declval<F>()(0))>
modal then(F f) {
    return [f = std::move(f)](int mode) { return static_cast<int>(f(mode)); };
}

inline modal then(int m) {
    return [m](int) { return m; };
}


// Option specification
// --------------------
//
// An option specification comprises zero or more keys (e.g. "-a", "--foo"),
// a sink (where the parsed argument will be sent), a parser - which may be
// the default parser - and zero or more flags that modify its behaviour.
//
// Option flags:

enum option_flag {
    flag = 1,       // Option takes no parameter.
    ephemeral = 2,  // Option is not saved in returned results.
    single = 4,     // Option is parsed at most once.
    mandatory = 8,  // Option must be present in argument list.
    exit = 16,      // Option stops further argument processing, return `nothing` from run().
    stop = 32,      // Option stops further argument processing, return saved options.
};

struct option {
    sink s;
    std::vector<key> keys;
    std::vector<filter> filters;
    std::vector<modal> modals;

    bool is_flag = false;
    bool is_ephemeral = false;
    bool is_single = false;
    bool is_mandatory = false;
    bool is_exit = false;
    bool is_stop = false;

    template <typename... Rest>
    option(sink s, Rest&&... rest): s(std::move(s)) {
        init_(std::forward<Rest>(rest)...);
    }

    bool has_key(const std::string& arg) const {
        if (keys.empty() && arg.empty()) return true;
        for (const auto& k: keys) {
            if (arg==k.label) return true;
        }
        return false;
    }

    bool check_mode(int mode) const {
        for (auto& f: filters) {
            if (!f(mode)) return false;
        }
        return true;
    }

    void set_mode(int& mode) const {
        for (auto& f: modals) mode = f(mode);
    }

    void run(const std::string& label, const char* arg) const {
        if (!is_flag && !arg) throw missing_argument(label);
        if (!s(arg)) throw option_parse_error(label);
    }

    std::string longest_label() const {
        const std::string* p = 0;
        for (auto& k: keys) {
            if (!p || k.label.size()>p->size()) p = &k.label;
        }
        return p? *p: std::string{};
    }

private:
    void init_() {}

    template <typename... Rest>
    void init_(enum option_flag f, Rest&&... rest) {
        is_flag      |= f & flag;
        is_ephemeral |= f & ephemeral;
        is_single    |= f & single;
        is_mandatory |= f & mandatory;
        is_exit      |= f & exit;
        is_stop      |= f & stop;
        init_(std::forward<Rest>(rest)...);
    }

    template <typename... Rest>
    void init_(filter f, Rest&&... rest) {
        filters.push_back(std::move(f));
        init_(std::forward<Rest>(rest)...);
    }

    template <typename... Rest>
    void init_(modal f, Rest&&... rest) {
        modals.push_back(std::move(f));
        init_(std::forward<Rest>(rest)...);
    }

    template <typename... Rest>
    void init_(key k, Rest&&... rest) {
        keys.push_back(std::move(k));
        init_(std::forward<Rest>(rest)...);
    }

};

// Saved options
// -------------
//
// Successfully matched options, excluding those with the to::ephemeral flag
// set, are collated in a saved_options structure for potential documentation
// or replay.

struct saved_options: private std::vector<std::string> {
    using std::vector<std::string>::begin;
    using std::vector<std::string>::end;
    using std::vector<std::string>::size;
    using std::vector<std::string>::empty;

    void add(std::string s) { push_back(std::move(s)); }

    saved_options& operator+=(const saved_options& so) {
        insert(end(), so.begin(), so.end());
        return *this;
    }

    struct arglist {
        int argc;
        char** argv;
        std::vector<char*> arg_data;
    };

    // Construct argv representing argument list.
    arglist as_arglist() const {
        arglist A;

        for (auto& a: *this) A.arg_data.push_back(const_cast<char*>(a.c_str()));
        A.arg_data.push_back(nullptr);
        A.argv = A.arg_data.data();
        A.argc = A.arg_data.size()-1;
        return A;
    }

    // Serialized representation:
    //
    // Saved arguments are separated by white space. If an argument
    // contains whitespace or a special character, it is escaped with
    // single quotes in a POSIX shell compatible fashion, so that
    // the representation can be used directly on a shell command line.

    friend std::ostream& operator<<(std::ostream& out, const saved_options& s) {
        auto escape = [](const std::string& v) {
            if (!v.empty() && v.find_first_of("\\*?[#~=%|^;<>()$'`\" \t\n")==std::string::npos) return v;

            // Wrap string in single quotes, replacing any internal single quote
            // character with: '\''

            std::string q ="'";
            for (auto c: v) {
                c=='\''? q += "'\\''": q += c;
            }
            return q += '\'';
        };

        bool first = true;
        for (auto& p: s) {
            if (first) first = false; else out << ' ';
            out << escape(p);
        }
        return out;
    }

    friend std::istream& operator>>(std::istream& in, saved_options& s) {
        std::string w;
        bool have_word = false;
        bool quote = false; // true => within single quotes.
        bool escape = false; // true => previous character was backslash.
        while (in) {
            char c = in.get();
            if (c==EOF) break;

            if (quote) {
                if (c!='\'') w += c;
                else quote = false;
            }
            else {
                if (escape) {
                    w += c;
                    escape = false;
                }
                else if (c=='\\') {
                    escape = true;
                    have_word = true;
                }
                else if (c=='\'') {
                    quote = true;
                    have_word = true;
                }
                else if (c!=' ' && c!='\t' && c!='\n') {
                    w += c;
                    have_word = true;
                }
                else {
                    if (have_word) s.add(w);
                    have_word = false;
                    w = "";
                }
            }
        }

        if (have_word) s.add(w);
        return in;
    }
};

// Option with mutable state (for checking single and mandatory flags),
// used by to::run().

namespace impl {
    struct counted_option: option {
        int count = 0;

        counted_option(const option& o): option(o) {}

        // On successful match, return pointers to matched key and value.
        // For flags, use nullptr for value; for empty key sets, use
        // nullptr for key.
        maybe<std::pair<const char*, const char*>> match(state& st) {
            if (is_flag) {
                for (auto& k: keys) {
                    if (st.match_flag(k)) return set(k.label, nullptr);
                }
                return nothing;
            }
            else if (!keys.empty()) {
                for (auto& k: keys) {
                    if (auto param = st.match_option(k)) return set(k.label, *param);
                }
                return nothing;
            }
            else {
                const char* param = *st.argv;
                st.shift();
                return set("", param);
            }
        }

        std::pair<const char*, const char*> set(const char* arg) {
            run("", arg);
            ++count;
            return {nullptr, arg};
        }

        std::pair<const char*, const char*> set(const std::string& label, const char* arg) {
            run(label, arg);
            ++count;
            return {label.c_str(), arg};
        }
    };
} // namespace impl

// Running a set of options
// ------------------------
//
// to::run() can be used to parse options from the command-line and/or from
// saved_options data.
//
// The first argument is a collection or sequence of option specifications,
// followed optionally by command line argc and argv or just argv. A
// saved_options object can be optionally passed as the last parameter.
//
// If an option with the to::exit flag is matched, option parsing will
// immediately stop and an empty value will be returned. Otherwise to::run()
// will return a saved_options structure recording the successfully parsed
// options.

namespace impl {
    inline maybe<saved_options> run(std::vector<impl::counted_option>& opts, int& argc, char** argv) {
        saved_options collate;
        bool exit = false;
        bool stop = false;
        state st{argc, argv};
        int mode = 0;
        while (st && !exit && !stop) {
            // Try options with a key first.
            for (auto& o: opts) {
                if (o.keys.empty()) continue;
                if (o.is_single && o.count) continue;
                if (!o.check_mode(mode)) continue;

                if (auto ma = o.match(st)) {
                    if (!o.is_ephemeral) {
                        if (ma->first) collate.add(ma->first);
                        if (ma->second) collate.add(ma->second);
                    }
                    o.set_mode(mode);
                    exit = o.is_exit;
                    stop = o.is_stop;
                    goto next;
                }
            }

            // Literal "--" terminates option parsing.
            if (!std::strcmp(*argv, "--")) {
                st.shift();
                return collate;
            }

            // Try free options.
            for (auto& o: opts) {
                if (!o.keys.empty()) continue;
                if (o.is_single && o.count) continue;
                if (!o.check_mode(mode)) continue;

                if (auto ma = o.match(st)) {
                    if (!o.is_ephemeral) collate.add(ma->second);
                    o.set_mode(mode);
                    exit = o.is_exit;
                    goto next;
                }
            }

            // Nothing matched, so increment argv.
            st.skip();
        next: ;
        }

        return exit? nothing: just(collate);
    }
} // namespace impl


template <typename Options>
maybe<saved_options> run(const Options& options, int& argc, char** argv, const saved_options& restore = saved_options{}) {
    using std::begin;
    using std::end;
    std::vector<impl::counted_option> opts(begin(options), end(options));
    auto r_args = restore.as_arglist();

    saved_options coll1, coll2;
    if (coll1 << impl::run(opts, r_args.argc, r_args.argv) && coll2 << impl::run(opts, argc, argv)) {
        for (auto& o: opts) {
            if (o.is_mandatory && !o.count) throw missing_mandatory_option(o.longest_label());
        }
        return coll1 += coll2;
    }

    return nothing;
}

template <typename Options>
maybe<saved_options> run(const Options& options, const saved_options& restore) {
    int ignore_argc = 0;
    char* end_of_args = nullptr;
    return run(options, ignore_argc, &end_of_args, restore);
}

template <typename Options>
maybe<saved_options> run(const Options& options, char** argv) {
    int ignore_argc = 0;
    return run(options, ignore_argc, argv);
}

template <typename Options>
maybe<saved_options> run(const Options& options, char** argv, const saved_options& restore) {
    int ignore_argc = 0;
    return run(options, ignore_argc, argv, restore);
}

// Running through command line arguments explicitly.
// --------------------------------------------------
//
// `to::parse<V>` checks the given argument against the provided keys, and on a match will try to
// parse and consume an argument of type V. A custom parser can be supplied.
//
// `to::parse<void>` will do the same, for flag options that take no argument.

template <
    typename V,
    typename P,
    typename = std::enable_if_t<!std::is_same<V, void>::value>,
    typename = std::enable_if_t<!std::is_convertible<const P&, key>::value>,
    typename... Tail
>
maybe<V> parse(char**& argp, const P& parser, key k0, Tail... krest) {
    key keys[] = { std::move(k0), std::move(krest)... };

    const char* arg = argp[0];
    if (!arg) return nothing;

    const char* text = nullptr;
    for (const key& k: keys) {
        if (k.label==arg) {
            if (!argp[1]) throw missing_argument(arg);
            text = argp[1];
            argp += 2;
            goto match;
        }
        else if (k.style==key::longfmt) {
            const char* eq = std::strrchr(arg, '=');
            if (eq && !std::strncmp(arg, k.label.c_str(), eq-arg)) {
                text = eq+1;
                argp += 1;
                goto match;
            }
        }
    }
    return nothing;

match:
    if (auto v = parser(text)) return v;
    else throw option_parse_error(arg);
}

template <
    typename V,
    typename = std::enable_if_t<!std::is_same<V, void>::value>,
    typename... Tail
>
maybe<V> parse(char**& argp, key k0, Tail... krest) {
    return parse<V>(argp, default_parser<V>{}, std::move(k0), std::move(krest)...);
}

template <typename... Tail>
maybe<void> parse(char**& argp, key k0, Tail... krest) {
    key keys[] = { std::move(k0), std::move(krest)... };

    const char* arg = argp[0];
    if (!arg) return nothing;

    for (const key& k: keys) {
        if (k.label==arg) {
            ++argp;
            return true;
        }
    }
    return nothing;
}

} // namespace to
