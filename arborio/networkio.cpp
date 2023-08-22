#include <any>
#include <limits>
#include <vector>

#include <arborio/networkio.hpp>

#include <arbor/common_types.hpp>
#include <arbor/network.hpp>
#include <arbor/util/expected.hpp>

#include "parse_helpers.hpp"

namespace arborio {

network_parse_error::network_parse_error(const std::string& msg, const arb::src_location& loc):
    arb::arbor_exception(
        concat("error in label description: ", msg, " at :", loc.line, ":", loc.column)) {}

namespace {
using eval_map_type = std::unordered_multimap<std::string, evaluator>;

eval_map_type network_eval_map{
    {"gid-range",
        make_call<int, int>([](int begin, int end) { return arb::gid_range(begin, end); },
            "Gid range [begin, end) with step size 1: ((begin:integer) (end:integer))")},
    {"gid-range",
        make_call<int, int, int>(
            [](int begin, int end, int step) { return arb::gid_range(begin, end, step); },
            "Gid range [begin, end) with step size: ((begin:integer) (end:integer) "
            "(step:integer))")},

    // cell kind
    {"cable-cell", make_call<>([]() { return arb::cell_kind::cable; }, "Cable cell kind")},
    {"lif-cell", make_call<>([]() { return arb::cell_kind::lif; }, "Lif cell kind")},
    {"benchmark-cell",
        make_call<>([]() { return arb::cell_kind::benchmark; }, "Benchmark cell kind")},
    {"spike-source-cell",
        make_call<>([]() { return arb::cell_kind::spike_source; }, "Spike source cell kind")},

    // network_selection
    {"all", make_call<>(arb::network_selection::all, "network selection of all cells and labels")},
    {"none", make_call<>(arb::network_selection::none, "network selection of no cells and labels")},
    {"inter-cell",
        make_call<>(arb::network_selection::inter_cell,
            "network selection of inter-cell connections only")},
    {"network-selection",
        make_call<std::string>(arb::network_selection::named,
            "network selection with 1 argument: (value:string)")},
    {"intersect",
        make_conversion_fold<arb::network_selection>(arb::network_selection::intersect,
            "intersection of network selections with at least 2 arguments: "
            "(network_selection network_selection [...network_selection])")},
    {"join",
        make_conversion_fold<arb::network_selection>(arb::network_selection::join,
            "join or union operation of network selections with at least 2 arguments: "
            "(network_selection network_selection [...network_selection])")},
    {"symmetric-difference",
        make_conversion_fold<arb::network_selection>(arb::network_selection::symmetric_difference,
            "symmetric difference operation between network selections with at least 2 arguments: "
            "(network_selection network_selection [...network_selection])")},
    {"difference",
        make_call<arb::network_selection, arb::network_selection>(
            arb::network_selection::difference,
            "difference of first selection with the second one: "
            "(network_selection network_selection)")},
    {"complement",
        make_call<arb::network_selection>(arb::network_selection::complement,
            "complement of given selection argument: (network_selection)")},
    {"source-cell-kind",
        make_call<arb::cell_kind>(arb::network_selection::source_cell_kind,
            "all sources of cells matching given cell kind argument: (kind:cell-kind)")},
    {"destination-cell-kind",
        make_call<arb::cell_kind>(arb::network_selection::destination_cell_kind,
            "all destinations of cells matching given cell kind argument: (kind:cell-kind)")},
    {"source-label",
        make_arg_vec_call<cell_tag_type>(
            [](const std::vector<std::variant<cell_tag_type>>& vec) {
                std::vector<cell_tag_type> labels;
                std::transform(
                    vec.begin(), vec.end(), std::back_inserter(labels), [](const auto& x) {
                        return std::get<cell_tag_type>(x);
                    });
                return arb::network_selection::source_label(std::move(labels));
            },
            "all sources in cell with gid in list: (gid:integer) [...(gid:integer)]")},
    {"destination-label",
        make_arg_vec_call<cell_tag_type>(
            [](const std::vector<std::variant<cell_tag_type>>& vec) {
                std::vector<cell_tag_type> labels;
                std::transform(
                    vec.begin(), vec.end(), std::back_inserter(labels), [](const auto& x) {
                        return std::get<cell_tag_type>(x);
                    });
                return arb::network_selection::destination_label(std::move(labels));
            },
            "all destinations in cell with gid in list: (gid:integer) [...(gid:integer)]")},
    {"source-cell",
        make_arg_vec_call<int>(
            [](const std::vector<std::variant<int>>& vec) {
                std::vector<cell_gid_type> gids;
                std::transform(vec.begin(), vec.end(), std::back_inserter(gids), [](const auto& x) {
                    return std::get<int>(x);
                });
                return arb::network_selection::source_cell(std::move(gids));
            },
            "all sources in cell with gid in list: (gid:integer) [...(gid:integer)]")},
    {"source-cell",
        make_call<arb::gid_range>(static_cast<arb::network_selection (*)(arb::gid_range)>(
                                      arb::network_selection::source_cell),
            "all sources in cell with gid range: (range:gid-range)")},
    {"destination-cell",
        make_arg_vec_call<int>(
            [](const std::vector<std::variant<int>>& vec) {
                std::vector<cell_gid_type> gids;
                std::transform(vec.begin(), vec.end(), std::back_inserter(gids), [](const auto& x) {
                    return std::get<int>(x);
                });
                return arb::network_selection::destination_cell(std::move(gids));
            },
            "all destinations in cell with gid in list: (gid:integer) [...(gid:integer)]")},
    {"destination-cell",
        make_call<gid_range>(static_cast<arb::network_selection (*)(arb::gid_range)>(
                                 arb::network_selection::destination_cell),
            "all destinations in cell with gid range: "
            "(range:gid-range)")},
    {"chain",
        make_arg_vec_call<int>(
            [](const std::vector<std::variant<int>>& vec) {
                std::vector<cell_gid_type> gids;
                std::transform(vec.begin(), vec.end(), std::back_inserter(gids), [](const auto& x) {
                    return std::get<int>(x);
                });
                return arb::network_selection::chain(std::move(gids));
            },
            "A chain of connections in the given order of gids in list, such that entry \"i\" is "
            "the source and entry \"i+1\" the destination: (gid:integer) [...(gid:integer)]")},
    {"chain",
        make_call<arb::gid_range>(
            static_cast<arb::network_selection (*)(arb::gid_range)>(arb::network_selection::chain),
            "A chain of connections for all gids in range [begin, end) with given step size. Each "
            "entry \"i\" is connected as source to the destination \"i+1\": (begin:integer) "
            "(end:integer) (step:integer)")},
    {"chain-reverse",
        make_call<arb::gid_range>(arb::network_selection::chain_reverse,
            "A chain of connections for all gids in range [begin, end) with given step size. Each "
            "entry \"i+1\" is connected as source to the destination \"i\". This results in "
            "connection directions in reverse compared to the (chain-range ...) selection: "
            "(begin:integer) "
            "(end:integer) (step:integer)")},
    {"random",
        make_call<int, double>(arb::network_selection::random,
            "randomly selected with given seed and probability. 2 arguments: (seed:integer, "
            "p:real)")},
    {"random",
        make_call<int, arb::network_value>(arb::network_selection::random,
            "randomly selected with given seed and probability function. Any probability value is "
            "clamped to [0.0, 1.0]. 2 arguments: (seed:integer, "
            "p:network-value)")},
    {"distance-lt",
        make_call<double>(arb::network_selection::distance_lt,
            "Select if distance between source and destination is less than given distance in "
            "micro meter: (distance:real)")},
    {"distance-gt",
        make_call<double>(arb::network_selection::distance_gt,
            "Select if distance between source and destination is greater than given distance in "
            "micro meter: (distance:real)")},

    // network_value
    {"scalar",
        make_call<double>(arb::network_value::scalar,
            "A fixed scalar value. 1 argument: (value:real)")},
    {"network-value",
        make_call<std::string>(arb::network_value::named,
            "A named network value with 1 argument: (value:string)")},
    {"distance",
        make_call<double>(arb::network_value::distance,
            "Distance between source and destination scaled by given value with unit [1/um]. 1 "
            "argument: (scale:real)")},
    {"distance",
        make_call<>([]() { return arb::network_value::distance(1.0); },
            "Distance between source and destination scaled by 1.0 with unit [1/um].")},
    {"uniform-distribution",
        make_call<int, double, double>(
            [](unsigned seed, double begin, double end) {
                return arb::network_value::uniform_distribution(seed, {begin, end});
            },
            "Uniform random distribution within interval [begin, end): (seed:integer, begin:real, "
            "end:real)")},
    {"normal-distribution",
        make_call<int, double, double>(arb::network_value::normal_distribution,
            "Normal random distribution with given mean and standard deviation: (seed:integer, "
            "mean:real, std_deviation:real)")},
    {"truncated-normal-distribution",
        make_call<int, double, double, double, double>(
            [](unsigned seed, double mean, double std_deviation, double begin, double end) {
                return arb::network_value::truncated_normal_distribution(
                    seed, mean, std_deviation, {begin, end});
            },
            "Truncated normal random distribution with given mean and standard deviation within "
            "interval [begin, end]: (seed:integer, mean:real, std_deviation:real, begin:real, "
            "end:real)")},
    {"if-else",
        make_call<network_selection, network_value, network_value>(arb::network_value::if_else,
            "Returns the first network-value if a connection is the given network-selection and "
            "the second network-value otherwise. 3 arguments: (sel:network-selection, "
            "true_value:network-value, false_value:network_value)")},
    {"add",
        make_conversion_fold<arb::network_value, arb::network_value, double>(
            arb::network_value::add,
            "Sum of network values with at least 2 arguments: ((network-value | double) "
            "(network-value | double) [...(network-value | double)])")},
    {"sub",
        make_conversion_fold<arb::network_value, arb::network_value, double>(
            arb::network_value::sub,
            "Subtraction of network values from the first argument with at least 2 arguments: "
            "((network-value | double) (network-value | double) [...(network-value | double)])")},
    {"mul",
        make_conversion_fold<arb::network_value, arb::network_value, double>(
            arb::network_value::mul,
            "Multiplication of network values with at least 2 arguments: ((network-value | double) "
            "(network-value | double) [...(network-value | double)])")},
    {"div",
        make_conversion_fold<arb::network_value, arb::network_value, double>(
            arb::network_value::div,
            "Division of the first argument by each following network value sequentially with at "
            "least 2 arguments: ((network-value | double) "
            "(network-value | double) [...(network-value | double)])")},
    {"min",
        make_conversion_fold<arb::network_value, arb::network_value, double>(
            arb::network_value::min,
            "Minimum of network values with at least 2 arguments: ((network-value | double) "
            "(network-value | double) [...(network-value | double)])")},
    {"max",
        make_conversion_fold<arb::network_value, arb::network_value, double>(
            arb::network_value::max,
            "Minimum of network values with at least 2 arguments: ((network-value | double) "
            "(network-value | double) [...(network-value | double)])")},
    {"log", make_call<double>(arb::network_value::log, "Logarithm. 1 argument: (value:real)")},
    {"log",
        make_call<network_value>(arb::network_value::log, "Logarithm. 1 argument: (value:real)")},
    {"exp", make_call<double>(arb::network_value::exp, "Exponential function. 1 argument: (value:real)")},
    {"exp",
        make_call<network_value>(arb::network_value::exp, "Exponential function. 1 argument: (value:real)")},
};

parse_network_hopefully<std::any> eval(const s_expr& e, const eval_map_type& map);

parse_network_hopefully<std::vector<std::any>> eval_args(const s_expr& e,
    const eval_map_type& map) {
    if (!e) return {std::vector<std::any>{}};  // empty argument list
    std::vector<std::any> args;
    for (auto& h: e) {
        if (auto arg = eval(h, map)) { args.push_back(std::move(*arg)); }
        else { return util::unexpected(std::move(arg.error())); }
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
        if (t == typeid(int)) return "integer";
        if (t == typeid(double)) return "real";
        if (t == typeid(arb::region)) return "region";
        if (t == typeid(arb::locset)) return "locset";
        return "unknown";
    };

    const auto nargs = args.size();
    std::string msg = concat("'", name, "' with ", nargs, "argument", nargs != 1u ? "s:" : ":");
    if (nargs) {
        msg += " (";
        bool first = true;
        for (auto& a: args) {
            msg += concat(first ? "" : " ", type_string(a.type()));
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
parse_network_hopefully<std::any> eval(const s_expr& e, const eval_map_type& map) {
    if (e.is_atom()) { return eval_atom<network_parse_error>(e); }
    if (e.head().is_atom()) {
        // This must be a function evaluation, where head is the function name, and
        // tail is a list of arguments.

        // Evaluate the arguments, and return error state if an error occurred.
        auto args = eval_args(e.tail(), map);
        if (!args) { return util::unexpected(args.error()); }

        // Find all candidate functions that match the name of the function.
        auto& name = e.head().atom().spelling;
        auto matches = map.equal_range(name);
        // Search for a candidate that matches the argument list.
        for (auto i = matches.first; i != matches.second; ++i) {
            if (i->second.match_args(*args)) {  // found a match: evaluate and return.
                return i->second.eval(*args);
            }
        }

        // Unable to find a match: try to return a helpful error message.
        const auto nc = std::distance(matches.first, matches.second);
        auto msg = concat("No matches for ",
            eval_description(name.c_str(), *args),
            "\n  There are ",
            nc,
            " potential candidates",
            nc ? ":" : ".");
        int count = 0;
        for (auto i = matches.first; i != matches.second; ++i) {
            msg += concat("\n  Candidate ", ++count, "  ", i->second.message);
        }
        return util::unexpected(network_parse_error(msg, location(e)));
    }

    return util::unexpected(network_parse_error(
        concat("'", e, "' is not either integer, real expression of the form (op <args>)"),
        location(e)));
}

}  // namespace

parse_network_hopefully<arb::network_selection> parse_network_selection_expression(
    const std::string& s) {
    if (auto e = eval(parse_s_expr(s), network_eval_map)) {
        if (e->type() == typeid(arb::network_selection)) {
            return {std::move(std::any_cast<arb::network_selection&>(*e))};
        }
        return util::unexpected(network_parse_error(concat("Invalid iexpr description: '", s)));
    }
    else { return util::unexpected(network_parse_error(std::string() + e.error().what())); }
}

parse_network_hopefully<arb::network_value> parse_network_value_expression(const std::string& s) {
    if (auto e = eval(parse_s_expr(s), network_eval_map)) {
        if (e->type() == typeid(arb::network_value)) {
            return {std::move(std::any_cast<arb::network_value&>(*e))};
        }
        return util::unexpected(network_parse_error(concat("Invalid iexpr description: '", s)));
    }
    else { return util::unexpected(network_parse_error(std::string() + e.error().what())); }
}

}  // namespace arborio
