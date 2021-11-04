Smaller, better? option parsing.

## Goals

This project constitutes yet another header-only option parsing library for C++.

Design goals:

* Header only.
* C++14 and C++17 compatible.
* Fairly short and easy to customize.

Features:

* Support 'short' and 'long' style options: `-v 3` and `--value=3`.
* Support 'compact' bunching of options: `-abc 3` vs `-a -b -c 3`.
* Save and restore options and arguments in a shell-compatible format,
  allowing e.g. `` program `cat previous-options` --foo=bar ``.
* Options can be interpreted modally.

Non-features:

* _Does not support multibyte encodings with shift sequences or wide character streams._
  This is due to laziness. But it does try to at least not break UTF-8.
* _Does not automatically generate help/usage text._
  What constitutes good help output is too specific to any given program.
* _Does not support optional or multiple arguments to an option._
  This is mainly due to problems of ambiguous parsing, though in a pinch this can
  be set up through the use of modal option parsing (see _Filters and Modals_ below).

The library actually provides two interfaces:
1. One can iterate through the command line arguments explicitly, testing them
   with `to::parse`. This precludes the use of compact-style options or modal parsing,
   but gives more control to the user code.
2. Or one can make a table of `to::option` specifications, and pass them to
   `to::run`, which will handle all the parsing itself.

## Simple examples

More examples are found in the `ex/` subdirectory.

Simple interface (`to::parse`) code for parsing options three options, one numeric,
one a keyword from a table, and one just a flag.
```
#include <string>
#include <utility>
#include <tinyopt/tinyopt.h>

const char* usage_str =
    "[OPTION]...\n"
    "\n"
    "  -n, --number=N       Specify N\n"
    "  -f, --function=FUNC  Perform FUNC, which is one of: one, two\n"
    "  -h, --help           Display usage information and exit\n";

int main(int argc, char** argv) {
    try {
        int n = 1, fn = 0;
        bool help = false;

        std::pair<const char*, int> functions[] = {
            { "one", 1 }, { "two", 2 }
        };

        for (auto arg = argv+1; *arg; ) {
            bool ok =
                help << to::parse(arg, "-h", "--help") ||
                n    << to::parse<int>(arg, "-n", "--number") ||
                fn   << to::parse<int>(arg, to::keywords(functions), "-f", "--function");

            if (!ok) throw to::option_error("unrecognized argument", *arg);
        }

        if (help) {
            to::usage(argv[0], usage_str);
            return 0;
        }

        if (n<1) throw to::option_error("N must be at least 1");
        if (fn<1) throw to::option_error("Require FUNC");

        // Do things with arguments:

        for (int i = 0; i<n; ++i) {
            std::cout << "Performing function #" << fn << "\n";
        }
    }
    catch (to::option_error& e) {
        to::usage_error(argv[0], usage_str, e.what());
        return 1;
    }
}
```

Equivalent `to::run` version.
```
#include <string>
#include <utility>
#include <tinyopt/tinyopt.h>

const char* usage_str =
    "[OPTION]...\n"
    "\n"
    "  -n, --number=N       Specify N\n"
    "  -f, --function=FUNC  Perform FUNC, which is one of: one, two\n"
    "  -h, --help           Display usage information and exit\n";

int main(int argc, char** argv) {
    try {
        int n = 1, fn = 0;

        std::pair<const char*, int> functions[] = {
            { "one", 1 }, { "two", 2 }
        };

        auto help = [argv0 = argv[0]] { to::usage(argv0, usage_str); };

        to::option opts[] = {
            { n, "-n", "--number" },
            { {fn, to::keywords(functions)}, "-f", "--function", to::mandatory },
            { to::action(help), to::flag, to::exit, "-h", "--help" }
        };

        if (!to::run(opts, argc, argv+1)) return 0;

        if (argv[1]) throw to::option_error("unrecogonized argument", argv[1]);
        if (n<1) throw to::option_error("N must be at least 1");

        // Do things with arguments:

        for (int i = 0; i<n; ++i) {
            std::cout << "Performing function #" << fn << "\n";
        }
    }
    catch (to::option_error& e) {
        to::usage_error(argv[0], usage_str, e.what());
        return 1;
    }
}
```


## Building

Tinyopt is a header-only library, but the supplied Makefile will build the unit
tests and examples.

The Makefile is designed to support out-of-tree building, and the recommended
approach is to create a build directory, symbolicly link the project Makefile
into the directory, and build from there. For example, to check out, build the
tests and examples, and then run the unit tests:
```
    % git clone git@github.com:halfflat/tinyopt
    % cd tinyopt
    % mkdir build
    % cd build
    % ln -s ../Makefile .
    % make
    % ./unit
```


## Documentation

All tinyopt code lives in the namespace `to`. This namespace is omitted in the
descriptions below.

### Common classes and helpers

#### `template <typename V> struct maybe`

`maybe<V>` is a simple work-alike for (some of) the C++17 `std::optional<V>` class.
A default constructed `maybe<V>` has no value, and evaluates to false in a `bool`
context; it will otherwise evaluate to true.

If `m` is an object of type `maybe<V>`, then `*m` evaluates to the contained value
if defined. `m.value()` does the same, but will throw an exception of type
`std::invalid_argument` if `m` does not contain a value.

The special value `nothing` is implicitly convertible to an empty `maybe<V>` for any `V`.
The expression `just(v)` function returns a `maybe<V>` holding the value `v`.

As a special case, `maybe<void>` simply maintains a has-value state; it will return
true in a `bool` context if has been initialized or assigned with any `maybe<V>`
that contains a value, or by any other value that is not `nothing`. `something`
is a pre-defined non-empty value of type `maybe<void>`.

`maybe<V>` values support basic monadic-like functionality via `operator<<`.
* If `x` is an lvalue and `m` is of type `maybe<U>`, then 
  `x << m` has type `maybe<V>` (`V` is the type of `x=*m`) and assigns `m.value()` to `x`
  if `m` has a value. In the case that `U` is `void`, then the value of `m` is taken
  to be `true`.
* If `f` is a function or function object with signature `V f(U)`, and `m` is of type `maybe<U>`, then
  `f << m` has type `maybe<V>` and contains `f(*m)` if `m` has a value.
* if `f` has signature `V f()`, and `m` is of type `maybe<U>` or `maybe<void>`, then
  `f << m` has type `maybe<V>` and contains `f()` if `m` has a value.

#### `option_error`

An exception class derived from `std::runtime_error`. It has two constructors:
* `option_error(const std::string&)` simply sets the what string to the argument.
* `option_error(const std::string& message, const std::string& arg)` sets the what string to
   the value of `arg+": "+mesage`.

The option parsers can throw exceptions derived from `option_error`, namely:
`option_parse_error`, `missing_mandatory_option`, and `missing_argument`.

#### `usage(const char *argv0, const std::string& usagemsg, const std::string& prefix = "Usage: ")`

Extract a program name from `argv0` (everything after the last '/' if present) and
print a message to standard out in the form "Usage: <program-name> <usagemsg>\n".
An alternative prefix to "Usage: " can be supplied optionally.

#### `usage_error(const char *argv0, const std::string& usagemsg, const std::string& error, const std::string& prefix = "Usage: ")`

Extract a program name from `argv0` (everything after the last '/' if present) and
print a message to standard error in the form
`<program-name>: <error>\nUsage: <program-name> <usagemsg>\n`.
An alternative prefix to "Usage: " can be supplied optionally.

### Parsers

A parser is a function or functional object with signature `maybe<X> (const char*)`
for some type `X`. They are used to try to convert a C-string argument into a value.

If no explicit parser is given to a the`parse` function or to an `option` specification,
the default parser `default_parser` is used, which will use `std::istream::operator>>`
to read the supplied argument.

Tinyopt supplies additional parsers:

* `keyword_parser<V>`

   Constructed from a table of key-value pairs, the `keyword_parser<V>` parser
   will return the first value found in the table with matching key, or `nothing`
   if there is no match.

   The `keywords(pairs)` function constructs a `keyword_parser<V>` object from the
   collection of keyword pairs `pairs`, where each element in the collection is
   a `std::pair`. The first component of each pair is used to construct the `std::string`
   key in the keyword table, and the second the value. The value type `V` is deduced
   from this second component.

* `delimited_parser<P>`

   The delimited parser uses another parser of type `P` to parse individual
   elements in a delimited sequence, and returns a `std::vector` of the
   corresponding values.

   The convenience constructor `delimited<V>(char delim = ',')` will make
   a `delimited_parser` using the default parser for `V` and delimiter
   `delim` (by default, a comma).

   `delimited(char delim, P&& parser)` is a convenience wrapper for
   `delimited_parser<P>::delimited_parser(delim, parser)`.

### Keys

Keys are how options are specified on the command line. They consist of
a string label and a style, which is one of `key::shortfmt`,
`key::longfmt`, or `key::compact`.

All options that take an argument will take that argument from the
next item in the argument list, and only options with a 'compact'
key can be combined together in a single argument.

An option with a 'long' key can additionally take its argument by
following the key with an equals sign and then the argument value.

As an example, let "-s" be a short option key, "--long" a long option key,
"-a" a compact option key for a flag, and "-b" a compact option key for
an option that takes a value. Then the follwing are equivalent ways
for specifying these options on a command line:

```
-s 1 --long 2 -a -b 3
```

```
-s 1 --long=2 -a -b3
```

```
-s 1 --long=2 -ab3
```

Keys can be constructed explicitly, implicitly from labels, or
from the use of string literal functions:

* `key(std::string label, enum key::style style)`, `key(const char* label, enum key::style style)`

   Make a key with given label and style.

* `key(std::string label)`, `key(const char* label)`

   Make a key with the given label. The style will be `key::shortfmt`, unless
   the label starts with a double dash "--". This constructor is implicit.

* `operator""_short`, `operator""_long`, `operator""_compact`.

   Make a key in the corresonding style from a string literal.

The string literal operators are included in an inline namespace `literals`
that can be included in user code via `using namespace to::literals`.

### Using `to::parse`

The Tinyopt `to::parse` functions compare a single command line argument
against one or more short- or long-style options, parsing any corresponding
option argument with the default or explicitly provided parser.

* `maybe<void> parse(char**& argp, key k, ...)`

   Attempt to parse an option with no argument (i.e. a flag) at `argp`, given
   by the option key `k` or subsequent. Returns an empty `maybe<void>` value
   if it fails to match any of the keys.

   If the match is successful, increment `argp` to point to the next argument.

* `maybe<V> parse(char**& argp, const P& parser, key k, ...)`<br/>
  `maybe<V> parse(char**& argp, key k, ...)`

   Attempt to parse an option with an argument to be interpreted as type `V`,
   matching the option against the key `k` or subsequent. If no `parser` is
   supplied, use the default parser for the type `V` to convert the option
   argument.

   If the match and value pase is successful, increment `argp` once or twice
   as required to advance to the next option.

The `parse` functions will throw `missing_argument` if no argument is found
for a non-flag option, and `option_parse_error` if the parser for an argument
returns `nothing`.

The monadic `maybe` return values allow straightforward chaining of multiple
`to::parse` calls in a single expression; see `ex/ex1-parse.cc` for an
example.

### Using `to::run`

The `to::run` function hands more control to the library for option parsing.
The basic workflow is:

1. The user code sets up a collection of `option` objects, each of which describe
   a command line option or flag, and how to handle the result of parsing it.
2. This collection is passed to the `run` function, along with `argc` and `argv`.
   `argv` is modified in place to remove matched options; anything remaining
   can be handled by the user code.
3. The `run` function returns a saved set of matched options that can, for example,
   be saved in program output for tracking processing steps, or in some file
   to allow for re-execution of the code with the same arguments.

An `option` describes one command line flag or option with an argument. It has
five components: a `sink`, that describes what to do with a successfully parsed
option; a set of `key`s, which are how the option is presented on the command line;
a sequence of `filter`s, which can limit the scope in which the option is valid;
a sequence of `modal`s, which change the modal state of the parser when the option
is matched; and finally a set of option flags that modify behaviour.

#### Sinks

A sink wraps a function that takes a `const char*` value, representing the
argument to an option, and returns a `bool`. A return value of `true` indicates
a successful parsing of the argument; `false` represents a parse error.

A sink has three constructors:
* `sink(sink::action_t, Action a)`

  `action_t` is a tag type, indicating that the second argument is a functional
  to be used directly as the wrapped function. `sink::action` is a value with
  this type for use in this constructor.

  This constructor is used by the `action` wrapper function described below.

* `sink(V& var, P parser)`

  Make a sink that uses the parser `P` (see above for a description of what
  constitutes a tinyopt parser) to get a value of type `V`, and write that
  value to `var`.

* `sink(V& var)`

  Equivalent to `sink(var, default_parser<V>{})`

If an option doesn't have any associated argument, i.e. it is a flag,
the `sink` object is passed `nullptr`.

The `action` wrapper function takes a nullary or unary function or functional
object, and optionally a parser for the function's argument. It returns a
`sink` object that applies the default or supplied parser object
and if successful, calls the function with the parsed value.

A special action wrapper is `error(const std::string& message)`, which will
throw a `user_option_error` with the given message.

#### Sink adaptors

The library supplies some convenience adaptors for making `sink`s for common
situations:

* `push_back(Container& c, P parser = P{})`

  Append parsed values to the container `c` using `Container::push_back`.
  The default parser is `default_parser<Container::value_type>`.

* `set(V& v, X value)`

   Set `v` to `value`, ignoring any argument.

* `set(V& v)`

   Set `v` to `true`, ignoring any argument.

* `increment(V& v, X delta)`

   Perform `v += delta`, ignoring any argument.

* `increment(V& v)`

   Perform `++v`, ignoring any argument.

#### Filters and modals

Options are by default always available for consideration. The `single`
flag described below provides one simple constraint on option matching; the
modal interfaces provide a more elaborate system should it be required.

Each option maintains a sequence of _filters_ and a sequence of _modals_.

A _filter_ is a `filter` object (an alias for `std::function<bool (int)>`)
that takes the current mode (an integer, by default zero) and returns `false`
if the option should not be considered for matching. Options will be ignored if
any of its filters return false.

A _modal_ is a `modal` object (an alias for `std::functional<int (int)>`)
that is passed the current mode value and returns the new mode value when the
option is successfully matched and parsed.

Filters can be made with the `when` adaptor. Given a functional object,
it will wrap it in a `filter`. Given an integer value, it will make
a `filter` that returns true only if the mode matches that value.
If `when` is given multiple arguments, it constructs a filter that is
the disjunction of filters constructed from each argument.

`then(f)` constructs a `modal` object wrapping the functional object `f`;
`then(int v)` constructs a `modal` that just returns the value `m`.

#### Flags

Option behaviour can be modified by supplying `enum option_flag` values:

* `flag` — Option takes no argument, i.e. it is a flag.
* `ephemeral` — Do not record this option in the saved data returned by `run()`.
* `single` — This option will be checked at most once, and then ignored for the remainder of option processing.
* `mandatory` — Throw an exception if this option does not appear in the command line arguments.
* `exit` — On successful parsing of this option, stop any further option processing and return `nothing` from `run()`.
* `stop` — On successful parsing of this option, stop any further option processing but return saved options as normal from `run()`.

These enum values are all powers of two and can be combined via bitwise or `|`.

When to use `exit`, and when to use `stop`? `exit` is intended to describe
the situation where the program should not proceed further with normal
processing; a standard use case for `exit` is for `--help` options, which
should cause the program to exit after printing help text. `stop`, on the
other hand, is used for options that indicate that no further special
argument processing should be performed; this corresponds to the common
convention of `--` on the command line indicating that all remaining
arguments should be interpreted as command line options.

#### Specifying an option

The `option` constructor takes a `sink` as the first argument, followed by
any number of keys, filters, modals, and flags in any order.

An `option` may have no keys at all — these will always match an item in the
command line argument list, and that item will be passed directly to the
option's sink.

Some example specifications:
```
    // Saves integer argument to variable 'n':
    int n = 0;
    to::option opt_n = { n, "-n" };

    // Flag '-v' or '--verbose' that increases verbosity level, but is not
    // kept in the returned list of saved options.
    int verbosity = 0;
    to::option opt_v = { to::increment(verbosity), "-v", "--verbose", to::flag, to::ephemeral };

    // Save vector of values from one argument of comma separated values, e.g.
    // -x 1,2,3,4,5:
    std::vector<int> xs;
    to::option opt_x = { {xs, to::delimited<int>()}, "-x" };

    // Save vector of values one by one, e.g.,
    // -k 1 -k 2 -k 3 -k 4 -k 5
    to::option opt_k = { to::push_back(xs), "-k" };

    // A 'help' flag that calls a help() function and stops further option processing.
    to::option opt_h = { to::action(help), to::flag, to::exit, "-h", "--help" };

    // A modal flag that sets the mode to 2, and a filtered option that only
    // applies when the mode is 2.
    enum mode { none = 0, list = 1, install = 2} prog_mode = none;
    to::option opt_m = { to::set(prog_mode, install), "install", to::then(install) };
    to::option opt_h = { to::action(install_help), to::flag, to::exit, "-h", "--help", to::when(install) };

    // Compact option keys using to::literals:
    using namespace to::literals;
    bool a = false, b = false, c = false;
    to::option flags[] = {
        { to::set(a), "-a"_compact, to::flag },
        { to::set(b), "-b"_compact, to::flag },
        { to::set(c), "-c"_compact, to::flag }
    };
```

#### Saved options

The `run()` function (see below) returns a value of type
`maybe<saved_options>`. The `saved_options` object holds a record of the
successfully parsed options. It wraps a `std::vector<std::string>` that has one
element per option key and argument, in the order they were matched (excluding
options with the `ephemeral` flag).

While the contents can be inspected via methods `begin()`, `end()`, `empty()`
and `size()`, it is primarily intended to support serialization. Overloads of
`operator<<` and `operator>>` will write and read a `saved_options` object to a
`std::ostream` or `std::istream` object respectively. The serialized format
uses POSIX shell-compatible escaping with backslashes and single quotes so that
they be incorporated directly on the command line in later program invocations.

#### Running a set of options

A command line argument list or `saved_options` object is run against a
collection of `option` specifications with `run()`. There are five overloads,
each of which returns a `saved_options` value in normal execution or `nothing`
if an option with the `exit` flag is matched.

In the following `Options` is any iterable collection of `option` values.

* `maybe<saved_options> run(const Options& options, int& argc, char** argv)`

   Parse the items in `argv` against the options provided in the first argument.
   Starting at the beginning of the `argv` list, options with keys are checked first,
   in the order they appear in `options`, followed by options without keys.

   Successfully parsed options are removed from the `argv` list in-place, and
   `argc` is adjusted accordingly.

* `maybe<saved_options> run(const Options& options, int& argc, char** argv, const saved_options& restore)`

   As for `run(options, argc, argv)`, but first run the options against the saved command line
   arguments in `restore`, and then again against `argv`.

   A mandatory option can be satisfied by the restore set or by the argv set.

* `maybe<saved_options> run(const Options& options, const saved_options& restore)`

   As for `run(options, argc, argv, restore)`, but with an empty argc/argv list.

* `maybe<saved_options> run(const Options& options, char** argv)`

   As for `run(options, argc, argv)`, but ignoring argc.

* `maybe<saved_options> run(const Options& options, char** argv, const saved_options& restore)`

   As for `run(options, argc, argv, restore)`, but ignoring argc.

Like the `to::parse` functions, the `run()` function can throw `missing_argument` or
`option_parse_error`. In addition, it will throw `missing_mandatory_option` if an option
marked with `mandatory` is not found during command line argument parsing.

Note that the arguments in `argv` are checked from the beginning; when calling `run` from within,
e.g the main function `int main(int argc, char** argv)`, one should pass `argv+1` to `run`
so as to avoid including the program name in `argv[0]`.
