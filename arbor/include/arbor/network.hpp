#pragma once

#include <arbor/cable_cell_param.hpp>
#include <arbor/common_types.hpp>
#include <arbor/export.hpp>
#include <arbor/morph/primitives.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>

namespace arb {

using network_hash_type = std::uint64_t;

struct ARB_SYMBOL_VISIBLE network_site_info {
    network_site_info() = default;

    network_site_info(cell_gid_type gid,
        cell_lid_type lid,
        cell_kind kind,
        std::string_view label,
        mlocation location,
        mpoint global_location);

    cell_gid_type gid;
    cell_lid_type lid;
    cell_kind kind;
    std::string_view label;
    mlocation location;
    mpoint global_location;
    network_hash_type hash;
};

struct network_selection_impl;

struct network_value_impl;

class ARB_SYMBOL_VISIBLE network_label_dict;

class ARB_SYMBOL_VISIBLE network_selection {
public:
    using custom_func_type =
        std::function<bool(const network_site_info& src, const network_site_info& dest)>;

    network_selection() { *this = network_selection::none(); }

    // Select all
    static network_selection all();

    // Select none
    static network_selection none();

    static network_selection named(std::string name);

    static network_selection source_cell_kind(cell_kind kind);

    static network_selection destination_cell_kind(cell_kind kind);

    static network_selection source_label(std::vector<cell_tag_type> labels);

    static network_selection destination_label(std::vector<cell_tag_type> labels);

    static network_selection source_gid(std::vector<cell_gid_type> gids);

    static network_selection source_gid(cell_gid_type gid_begin, cell_gid_type gid_end);

    static network_selection destination_gid(std::vector<cell_gid_type> gids);

    static network_selection destination_gid(cell_gid_type gid_begin, cell_gid_type gid_end);

    static network_selection ring(std::vector<cell_gid_type> gids);

    static network_selection ring(cell_gid_type gid_begin, cell_gid_type gid_end);

    static network_selection intersect(network_selection left, network_selection right);

    static network_selection join(network_selection left, network_selection right);

    static network_selection symmetric_difference(network_selection left, network_selection right);

    static network_selection difference(network_selection left, network_selection right);

    // Invert the selection
    static network_selection complement(network_selection s);

    // Only select connections between different cells
    static network_selection inter_cell();

    // Random selection using the bernoulli random distribution with probability "p" between 0.0
    // and 1.0
    static network_selection random_bernoulli(unsigned seed, double p);

    // Custom selection using the provided function "func". Repeated calls with the same arguments
    // to "func" must yield the same result. For gap junction selection,
    // "func" must be symmetric (func(a,b) = func(b,a)).
    static network_selection custom(custom_func_type func);

    // only select within given distance. This may enable more efficient sampling through an
    // internal spatial data structure.
    static network_selection distance_lt(double distance);

    // only select if distance greater then given distance. This may enable more efficient sampling
    // through an internal spatial data structure.
    static network_selection distance_gt(double distance);

    // randomly selected with a probability linearly interpolated between [p_begin, p_end] based on
    // the distance in the interval [distance_begin, distance_end].
    static network_selection random_linear_distance(unsigned seed,
        double distance_begin,
        double p_begin,
        double distance_end,
        double p_end);

private:
    network_selection(std::shared_ptr<network_selection_impl> impl);

    friend std::shared_ptr<network_selection_impl> thingify(network_selection s,
        const network_label_dict& dict);

    std::shared_ptr<network_selection_impl> impl_;
};

class ARB_SYMBOL_VISIBLE network_value {
public:
    using custom_func_type =
        std::function<double(const network_site_info& src, const network_site_info& dest)>;

    network_value() { *this = network_value::scalar(0.0); }

    // Scalar value with conversion from double
    network_value(double value) { *this = network_value::scalar(value); }

    // Scalar value. Will always return the same value given at construction.
    static network_value scalar(double value);

    static network_value named(std::string name);

    // Uniform random value in (range[0], range[1]]. Always returns the same value for repeated
    // calls with the same arguments and calls are symmetric v(a, b) = v(b, a).
    static network_value uniform_distribution(unsigned seed, const std::array<double, 2>& range);

    // Radom value from a normal distribution with given mean and standard deviation. Always returns
    // the same value for repeated calls with the same arguments and calls are symmetric v(a, b) =
    // v(b, a).
    static network_value normal_distribution(unsigned seed, double mean, double std_deviation);

    // Radom value from a truncated normal distribution with given mean and standard deviation (of a
    // non-truncated normal distribution), where the value is always in (range[0], range[1]]. Always
    // returns the same value for repeated calls with the same arguments and calls are symmetric
    // v(a, b) = v(b, a). Note: Values are generated by reject-accept method from a normal
    // distribution. Low acceptance rate can leed to poor performance, for example with very small
    // ranges or a mean far outside the range.
    static network_value truncated_normal_distribution(unsigned seed,
        double mean,
        double std_deviation,
        const std::array<double, 2>& range);

    // Custom value using the provided function "func". Repeated calls with the same arguments
    // to "func" must yield the same result. For gap junction values,
    // "func" must be symmetric (func(a,b) = func(b,a)).
    static network_value custom(custom_func_type func);

private:
    network_value(std::shared_ptr<network_value_impl> impl);

    friend std::shared_ptr<network_value_impl> thingify(network_value v,
        const network_label_dict& dict);

    std::shared_ptr<network_value_impl> impl_;
};

class ARB_SYMBOL_VISIBLE network_label_dict {
public:
    using ns_map = std::unordered_map<std::string, network_selection>;
    using nv_map = std::unordered_map<std::string, network_value>;

    network_label_dict& set(const std::string& name, network_selection s);

    network_label_dict& set(const std::string& name, network_value v);

    std::optional<network_selection> selection(const std::string& name) const;

    std::optional<network_value> value(const std::string& name) const;

    inline const ns_map& selections() const { return selections_; }

    inline const nv_map& values() const { return values_; }

private:
    ns_map selections_;
    nv_map values_;
};


struct network_description {
    network_selection selection;
    network_value weight;
    network_value delay;
    network_label_dict dict;
};

}  // namespace arb
