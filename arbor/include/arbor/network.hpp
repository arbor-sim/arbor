#pragma once

#include <arbor/common_types.hpp>
#include <arbor/export.hpp>
#include <arbor/recipe.hpp>
#include <arbor/cable_cell_param.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace arb {

using network_location = std::array<double, 3>;

using network_hash_type = std::uint64_t;

class network_site_selection {
public:
    static network_site_selection all();

    static network_site_selection none();

    static network_site_selection has_cell_kind(cell_kind kind);

    static network_site_selection has_tag(std::vector<cell_tag_type> tags);

    static network_site_selection has_gid(std::vector<cell_gid_type> gids);

    static network_site_selection has_gid_in_range(cell_gid_type begin, cell_gid_type end);

    static network_site_selection invert(network_site_selection s);

    network_site_selection operator&(network_site_selection right) const;

    network_site_selection operator|(network_site_selection right) const;

    network_site_selection operator^(network_site_selection right) const;

    bool operator()(cell_gid_type gid,
        cell_kind kind,
        const cell_tag_type& tag,
        const mlocation& local_loc,
        const network_location& global_loc) const;

private:
};

class ARB_SYMBOL_VISIBLE network_connection_selection {
public:
    // Random selection using the bernoulli random distribution with probability "p" between 0.0
    // and 1.0
    static network_connection_selection bernoulli_random(unsigned seed, double p);

    // Custom selection using the provided function "func". Repeated calls with the same arguments
    // to "func" must yield the same result. For gap junction selection,
    // "func" must be symmetric (func(a,b) = func(b,a)).
    static network_connection_selection custom(std::function<
        bool(cell_gid_type, const network_location&, cell_gid_type, const network_location&)> func);

    // Select all
    static network_connection_selection all();

    // Select none
    static network_connection_selection none();

    // Invert the selection
    static network_connection_selection invert(network_connection_selection s);

    // Only select connections between different cells
    static network_connection_selection inter_cell();

    // Only select connections when the global labels are not equal. May select intra-cell
    // connections, if the local label is not equal.
    static network_connection_selection not_equal();

    // only select within given distance. This may enable more efficient sampling through an
    // internal spatial data structure.
    static network_connection_selection within_distance(double distance);

    // random bernoulli sampling with a linear interpolated probabilty based on distance. Returns
    // "false" for any distance outside of the interval [distance_begin, distance_end].
    static network_connection_selection linear_bernoulli_random(unsigned seed,
        double distance_begin,
        double p_begin,
        double distance_end,
        double p_end);

    // Returns true if a connection between src and dest is selected.
    bool operator()(const cell_global_label_type& src,
        const network_location& src_location,
        const cell_global_label_type& dest,
        const network_location& dest_location) const;

    network_connection_selection operator&(network_connection_selection right) const;

    network_connection_selection operator|(network_connection_selection right) const;

    network_connection_selection operator^(network_connection_selection right) const;

    // Returns true if a connection between src and dest is selected.
    inline bool operator()(cell_gid_type src_gid,
        const network_location& global_src_location,
        network_hash_type src_hash,
        cell_gid_type dest_gid,
        const network_location& global_dest_location,
        network_hash_type dest_hash) const {
        return impl_->select(
            src_gid, global_src_location, src_hash, dest_gid, global_dest_location, dest_hash);
    }

    inline std::optional<double> max_distance() const { return impl_->max_distance(); }

private:
    struct selection_impl {
        virtual bool select(cell_gid_type src_gid,
            const network_location& global_src_location,
            network_hash_type src_hash,
            cell_gid_type dest_gid,
            const network_location& global_dest_location,
            network_hash_type dest_hash) const = 0;

        virtual std::optional<double> max_distance() const { return std::nullopt; }

        virtual ~selection_impl() = default;
    };

    struct bernoulli_random_impl;
    struct custom_impl;
    struct inter_cell_impl;
    struct not_equal_impl;
    struct all_impl;
    struct none_impl;
    struct and_impl;
    struct or_impl;
    struct xor_impl;
    struct invert_impl;
    struct within_distance_impl;
    struct linear_bernoulli_random_impl;

    network_connection_selection(std::shared_ptr<selection_impl> impl);

    inline bool select(cell_gid_type src_gid,
            const network_location& global_src_location,
            network_hash_type src_hash,
            cell_gid_type dest_gid,
            const network_location& global_dest_location,
            network_hash_type dest_hash) const {
        return impl_->select(
            src_gid, global_src_location, src_hash, dest_gid, global_dest_location, dest_hash);
    }

    std::shared_ptr<selection_impl> impl_;
};

class ARB_SYMBOL_VISIBLE network_value {
public:
    // Uniform value
    network_value(double value);

    // Uniform value. Will always return the same value given at construction.
    static network_value uniform(double value);

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
    static network_value custom(std::function<double(const cell_global_label_type&,
            const network_location&,
            const cell_global_label_type&,
            const network_location&,
            double)> func);

    inline double operator()(cell_gid_type src_gid,
        const network_location& global_src_location,
        network_hash_type src_hash,
        cell_gid_type dest_gid,
        const network_location& global_dest_location,
        network_hash_type dest_hash) const {
        return impl_->get(
            src_gid, global_src_location, src_hash, dest_gid, global_dest_location, dest_hash);
    }

private:

    struct value_impl {
        virtual double get(cell_gid_type src_gid,
        const network_location& global_src_location,
        network_hash_type src_hash,
        cell_gid_type dest_gid,
        const network_location& global_dest_location,
        network_hash_type dest_hash) const = 0;

        virtual ~value_impl() = default;
    };

    struct uniform_distribution_impl;
    struct normal_distribution_impl;
    struct truncated_normal_distribution_impl;
    struct custom_impl;
    struct uniform_impl;

    network_value(std::shared_ptr<value_impl> impl);

    inline double get(cell_gid_type src_gid,
        const network_location& global_src_location,
        network_hash_type src_hash,
        cell_gid_type dest_gid,
        const network_location& global_dest_location,
        network_hash_type dest_hash) const {
        return impl_->get(
            src_gid, global_src_location, src_hash, dest_gid, global_dest_location, dest_hash);
    }

    std::shared_ptr<value_impl> impl_;
};

struct network_description {
    network_site_selection src_selection;
    network_site_selection dest_selection;
    network_connection_selection connection_selection;
    network_value weight;
    network_value delay;
};

}  // namespace arb
