#include <arbor/common_types.hpp>
#include <arbor/network.hpp>

#include <Random123/threefry.h>
#include <Random123/boxmuller.hpp>
#include <Random123/uniform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "network_impl.hpp"

namespace arb {

namespace {

// Partial seed to use for network_value and network_selection generation.
// Different seed for each type to avoid unintentional correlation.
enum class network_seed : unsigned {
    selection_bernoulli = 2058443,
    selection_linear_bernoulli = 839033,
    value_uniform = 48202,
    value_normal = 8405,
    value_truncated_normal = 380237,
    site_info = 984293
};

// We only need minimal hash collisions and good spread over the hash range, because this will be
// used as input for random123, which then provides all desired hash properties.
// std::hash is implementation dependent, so we define our own for reproducibility.

std::uint64_t simple_string_hash(const std::string_view& s) {
    // use fnv1a hash algorithm
    constexpr std::uint64_t prime = 1099511628211ull;
    std::uint64_t h = 14695981039346656037ull;

    for (auto c: s) {
        h ^= c;
        h *= prime;
    }

    return h;
}

double uniform_rand_from_key_pair(std::array<unsigned, 2> seed,
    network_hash_type key_a,
    network_hash_type key_b) {
    using rand_type = r123::Threefry2x64;
    const rand_type::ctr_type seed_input = {{seed[0], seed[1]}};

    const rand_type::key_type key = {{std::min(key_a, key_b), std::max(key_a, key_b)}};
    rand_type gen;
    return r123::u01<double>(gen(seed_input, key)[0]);
}

double normal_rand_from_key_pair(std::array<unsigned, 2> seed,
    std::uint64_t key_a,
    std::uint64_t key_b) {
    using rand_type = r123::Threefry2x64;
    const rand_type::ctr_type seed_input = {{seed[0], seed[1]}};

    const rand_type::key_type key = {{std::min(key_a, key_b), std::max(key_a, key_b)}};
    rand_type gen;
    const auto rand_num = gen(seed_input, key);
    return r123::boxmuller(rand_num[0], rand_num[1]).x;
}


double network_location_distance(const network_location& a, const network_location& b) {
    return std::sqrt(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

struct network_selection_all_impl: public network_selection_impl {
    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return true;
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }
};

struct network_selection_none_impl: public network_selection_impl {

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return false;
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return false;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return false;
    }
};

struct network_selection_source_cell_kind_impl: public network_selection_impl {
    cell_kind select_kind;

    explicit network_selection_source_cell_kind_impl(cell_kind k): select_kind(k) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return src.kind == select_kind;
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return kind == select_kind;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }
};

struct network_selection_destination_cell_kind_impl: public network_selection_impl {
    cell_kind select_kind;

    explicit network_selection_destination_cell_kind_impl(cell_kind k): select_kind(k) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return src.kind == select_kind;
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return kind == select_kind;
    }
};

struct network_selection_source_label_impl: public network_selection_impl {
    std::vector<cell_tag_type> sorted_labels;

    explicit network_selection_source_label_impl(std::vector<cell_tag_type> labels):
        sorted_labels(std::move(labels)) {
        std::sort(sorted_labels.begin(), sorted_labels.end());
    }

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return std::binary_search(sorted_labels.begin(), sorted_labels.end(), src.label);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return std::binary_search(sorted_labels.begin(), sorted_labels.end(), label);
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }
};

struct network_selection_destination_label_impl: public network_selection_impl {
    std::vector<cell_tag_type> sorted_labels;

    explicit network_selection_destination_label_impl(std::vector<cell_tag_type> labels):
        sorted_labels(std::move(labels)) {
        std::sort(sorted_labels.begin(), sorted_labels.end());
    }

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return std::binary_search(sorted_labels.begin(), sorted_labels.end(), dest.label);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return std::binary_search(sorted_labels.begin(), sorted_labels.end(), label);
    }
};

struct network_selection_source_gid_impl: public network_selection_impl {
    std::vector<cell_gid_type> sorted_gids;

    explicit network_selection_source_gid_impl(std::vector<cell_gid_type> gids):
        sorted_gids(std::move(gids)) {
        std::sort(sorted_gids.begin(), sorted_gids.end());
    }

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return std::binary_search(sorted_gids.begin(), sorted_gids.end(), src.gid);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return std::binary_search(sorted_gids.begin(), sorted_gids.end(), gid);
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }
};

struct network_selection_destination_gid_impl: public network_selection_impl {
    std::vector<cell_gid_type> sorted_gids;

    explicit network_selection_destination_gid_impl(std::vector<cell_gid_type> gids):
        sorted_gids(std::move(gids)) {
        std::sort(sorted_gids.begin(), sorted_gids.end());
    }

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return std::binary_search(sorted_gids.begin(), sorted_gids.end(), dest.gid);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return std::binary_search(sorted_gids.begin(), sorted_gids.end(), gid);
    }
};

struct network_selection_invert_impl: public network_selection_impl {
    std::shared_ptr<network_selection_impl> selection;

    explicit network_selection_invert_impl(std::shared_ptr<network_selection_impl> s):
        selection(std::move(s)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return !selection->select_connection(src, dest);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;  // cannot exclude any because source selection cannot be inverted without
                      // knowing selection criteria.
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;  // cannot exclude any because destination selection cannot be inverted without
                      // knowing selection criteria.
    }
};

struct network_selection_inter_cell_impl: public network_selection_impl {
    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return src.gid != dest.gid;
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }
};

struct network_selection_not_equal_impl: public network_selection_impl {
    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return src.gid != dest.gid || src.label != dest.label || src.location != dest.location;
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }
};

struct network_selection_custom_impl: public network_selection_impl {
    std::function<bool(const network_site_info& src, const network_site_info& dest)> func;

    explicit network_selection_custom_impl(
        std::function<bool(const network_site_info& src, const network_site_info& dest)> f):
        func(std::move(f)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return func(src, dest);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }
};

struct network_selection_within_distance_impl: public network_selection_impl {
    double distance;

    explicit network_selection_within_distance_impl(double distance): distance(distance) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return network_location_distance(src.global_location, dest.global_location) <= distance;
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    std::optional<double> max_distance() const override { return distance; }
};

struct network_selection_bernoulli_random_impl: public network_selection_impl {
    unsigned seed;
    double probability;

    network_selection_bernoulli_random_impl(unsigned seed, double p): seed(seed), probability(p) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return uniform_rand_from_key_pair({unsigned(network_seed::selection_bernoulli), seed},
                   src.hash,
                   dest.hash) < probability;
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }
};

struct network_selection_linear_bernoulli_random_impl: public network_selection_impl {
    unsigned seed;
    double distance_begin;
    double p_begin;
    double distance_end;
    double p_end;

    network_selection_linear_bernoulli_random_impl(unsigned seed_,
        double distance_begin_,
        double p_begin_,
        double distance_end_,
        double p_end_):
        seed(seed_),
        distance_begin(distance_begin_),
        p_begin(p_begin_),
        distance_end(distance_end_),
        p_end(p_end_) {
        if (distance_begin > distance_end) {
            std::swap(distance_begin, distance_end);
            std::swap(p_begin, p_end);
        }
    }

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        const double distance = network_location_distance(src.global_location, dest.global_location);

        if (distance < distance_begin || distance > distance_end) return false;

        const double p =
            (p_begin * (distance_end - distance) + p_end * (distance - distance_begin)) /
            (distance_end - distance_begin);

        return uniform_rand_from_key_pair(
                   {unsigned(network_seed::selection_linear_bernoulli), seed},
                   src.hash,
                   dest.hash) < p;
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    std::optional<double> max_distance() const override { return distance_end; }
};

struct network_selection_and_impl: public network_selection_impl {
    std::shared_ptr<network_selection_impl> left, right;

    network_selection_and_impl(std::shared_ptr<network_selection_impl> l,
        std::shared_ptr<network_selection_impl> r):
        left(std::move(l)),
        right(std::move(r)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return left->select_connection(src, dest) && right->select_connection(src, dest);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return left->select_source(kind, gid, label) && right->select_source(kind, gid, label);
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return left->select_destination(kind, gid, label) &&
               right->select_destination(kind, gid, label);
    }

    std::optional<double> max_distance() const override {
        const auto d_left = left->max_distance();
        const auto d_right = right->max_distance();

        if (d_left && d_right) return std::min(d_left.value(), d_right.value());
        if (d_left) return d_left.value();
        if (d_right) return d_right.value();

        return std::nullopt;
    }
};

struct network_selection_or_impl: public network_selection_impl {
    std::shared_ptr<network_selection_impl> left, right;

    network_selection_or_impl(std::shared_ptr<network_selection_impl> l,
        std::shared_ptr<network_selection_impl> r):
        left(std::move(l)),
        right(std::move(r)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return left->select_connection(src, dest) || right->select_connection(src, dest);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return left->select_source(kind, gid, label) || right->select_source(kind, gid, label);
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return left->select_destination(kind, gid, label) ||
               right->select_destination(kind, gid, label);
    }

    std::optional<double> max_distance() const override {
        const auto d_left = left->max_distance();
        const auto d_right = right->max_distance();

        if (d_left && d_right) return std::max(d_left.value(), d_right.value());

        return std::nullopt;
    }
};

struct network_selection_xor_impl: public network_selection_impl {
    std::shared_ptr<network_selection_impl> left, right;

    network_selection_xor_impl(std::shared_ptr<network_selection_impl> l,
        std::shared_ptr<network_selection_impl> r):
        left(std::move(l)),
        right(std::move(r)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return left->select_connection(src, dest) ^ right->select_connection(src, dest);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;
    }

    std::optional<double> max_distance() const override {
        const auto d_left = left->max_distance();
        const auto d_right = right->max_distance();

        if (d_left && d_right) return std::max(d_left.value(), d_right.value());

        return std::nullopt;
    }
};


struct network_value_uniform_impl : public network_value_impl{
    double value;

    network_value_uniform_impl(double v): value(v) {}

    double get(const network_site_info& src, const network_site_info& dest) const override {
        return value;
    }
};

struct network_value_uniform_distribution_impl : public network_value_impl{
    unsigned seed = 0;
    std::array<double, 2> range;

    network_value_uniform_distribution_impl(unsigned rand_seed, const std::array<double, 2>& r):
        seed(rand_seed),
        range(r) {
        if (range[0] >= range[1])
            throw std::invalid_argument("Uniform distribution: invalid range");
    }

    double get(const network_site_info& src, const network_site_info& dest) const override {
        if (range[0] > range[1]) return range[1];

        // random number between 0 and 1
        const auto rand_num = uniform_rand_from_key_pair(
            {unsigned(network_seed::value_uniform), seed}, src.hash, dest.hash);

        return (range[1] - range[0]) * rand_num + range[0];
    }
};

struct network_value_normal_distribution_impl: public network_value_impl {
    unsigned seed = 0;
    double mean = 0.0;
    double std_deviation = 1.0;

    network_value_normal_distribution_impl(unsigned rand_seed, double mean_, double std_deviation_):
        seed(rand_seed),
        mean(mean_),
        std_deviation(std_deviation_) {}

    double get(const network_site_info& src, const network_site_info& dest) const override {
        return mean + std_deviation *
                          normal_rand_from_key_pair(
                              {unsigned(network_seed::value_normal), seed}, src.hash, dest.hash);
    }
};

struct network_value_truncated_normal_distribution_impl: public network_value_impl {
    unsigned seed = 0;
    double mean = 0.0;
    double std_deviation = 1.0;
    std::array<double, 2> range;

    network_value_truncated_normal_distribution_impl(unsigned rand_seed,
        double mean_,
        double std_deviation_,
        const std::array<double, 2>& range_):
        seed(rand_seed),
        mean(mean_),
        std_deviation(std_deviation_),
        range(range_) {
        if (range[0] >= range[1])
            throw std::invalid_argument("Truncated normal distribution: invalid range");
    }

    double get(const network_site_info& src, const network_site_info& dest) const override {

        const auto src_hash = src.hash;
        auto dest_hash = dest.hash;

        double value = 0.0;

        do {
            value =
                mean + std_deviation * normal_rand_from_key_pair(
                                           {unsigned(network_seed::value_truncated_normal), seed},
                                           src_hash,
                                           dest_hash);
            ++dest_hash;
        } while (!(value > range[0] && value <= range[1]));

        return value;
    }
};

struct network_value_custom_impl: public network_value_impl {
    std::function<double(const network_site_info& src, const network_site_info& dest)> func;

    network_value_custom_impl(
        std::function<double(const network_site_info& src, const network_site_info& dest)> f):
        func(std::move(f)) {}

    inline double get(const network_site_info& src, const network_site_info& dest) const override {
        return func(src, dest);
    }
};

}  // namespace

network_site_info::network_site_info(cell_gid_type gid,
    cell_lid_type lid,
    cell_kind kind,
    std::string_view label,
    mlocation location,
    network_location global_location):
    gid(gid),
    lid(lid),
    kind(kind),
    label(std::move(label)),
    location(location),
    global_location(global_location) {

    std::uint64_t label_hash = simple_string_hash(this->label);
    static_assert(sizeof(decltype(mlocation::pos)) == sizeof(std::uint64_t));
    std::uint64_t loc_pos_hash = *reinterpret_cast<const std::uint64_t*>(&location.pos);

    const auto seed = static_cast<std::uint64_t>(network_seed::site_info);

    using rand_type = r123::Threefry4x64;
    const rand_type::ctr_type seed_input = {{seed, 2 * seed, 3 * seed, 4 * seed}};
    const rand_type::key_type key = {{gid, label_hash, location.branch, loc_pos_hash}};

    rand_type gen;
    hash = gen(seed_input, key)[0];
}

network_selection::network_selection(std::shared_ptr<network_selection_impl> impl):
    impl_(std::move(impl)) {}

network_selection network_selection::operator&(network_selection right) const {
    return network_selection(
        std::make_shared<network_selection_and_impl>(this->impl_, std::move(right.impl_)));
}

network_selection network_selection::operator|(network_selection right) const {
    return network_selection(
        std::make_shared<network_selection_or_impl>(this->impl_, std::move(right.impl_)));
}

network_selection network_selection::operator^(network_selection right) const {
    return network_selection(
        std::make_shared<network_selection_xor_impl>(this->impl_, std::move(right.impl_)));
}

network_selection network_selection::all() {
    return network_selection(std::make_shared<network_selection_all_impl>());
}

network_selection network_selection::none() {
    return network_selection(std::make_shared<network_selection_none_impl>());
}

network_selection network_selection::source_cell_kind(cell_kind kind) {
    return network_selection(std::make_shared<network_selection_source_cell_kind_impl>(kind));
}

network_selection network_selection::destination_cell_kind(cell_kind kind) {
    return network_selection(std::make_shared<network_selection_destination_cell_kind_impl>(kind));
}

network_selection network_selection::source_label(std::vector<cell_tag_type> labels) {
    return network_selection(std::make_shared<network_selection_source_label_impl>(std::move(labels)));
}

network_selection network_selection::destination_label(std::vector<cell_tag_type> labels) {
    return network_selection(std::make_shared<network_selection_destination_label_impl>(std::move(labels)));
}

network_selection network_selection::source_gid(std::vector<cell_gid_type> gids) {
    return network_selection(std::make_shared<network_selection_source_gid_impl>(std::move(gids)));
}

network_selection network_selection::destination_gid(std::vector<cell_gid_type> gids) {
    return network_selection(std::make_shared<network_selection_destination_gid_impl>(std::move(gids)));
}

network_selection network_selection::invert(network_selection s) {
    return network_selection(std::make_shared<network_selection_invert_impl>(std::move(s.impl_)));
}

network_selection network_selection::inter_cell() {
    return network_selection(std::make_shared<network_selection_inter_cell_impl>());
}

network_selection network_selection::not_equal() {
    return network_selection(std::make_shared<network_selection_not_equal_impl>());
}

network_selection network_selection::bernoulli_random(unsigned seed, double p) {
    return network_selection(std::make_shared<network_selection_bernoulli_random_impl>(seed, p));
}

network_selection network_selection::custom(
    std::function<bool(const network_site_info& src, const network_site_info& dest)> func) {
    return network_selection(std::make_shared<network_selection_custom_impl>(std::move(func)));
}

network_selection network_selection::within_distance(double distance) {
    return network_selection(std::make_shared<network_selection_within_distance_impl>(distance));
}

network_selection network_selection::linear_bernoulli_random(unsigned seed,
    double distance_begin,
    double p_begin,
    double distance_end,
    double p_end) {
    return network_selection(std::make_shared<network_selection_linear_bernoulli_random_impl>(
        seed, distance_begin, p_begin, distance_end, p_end));
}

network_value::network_value(std::shared_ptr<network_value_impl> impl): impl_(std::move(impl)) {}

network_value network_value::uniform(double value) {
    return network_value(std::make_shared<network_value_uniform_impl>(value));
}

network_value network_value::uniform_distribution(unsigned seed,
    const std::array<double, 2>& range) {
    return network_value(std::make_shared<network_value_uniform_distribution_impl>(seed, range));
}

network_value network_value::normal_distribution(unsigned seed, double mean, double std_deviation) {
    return network_value(
        std::make_shared<network_value_normal_distribution_impl>(seed, mean, std_deviation));
}

network_value network_value::truncated_normal_distribution(unsigned seed,
    double mean,
    double std_deviation,
    const std::array<double, 2>& range) {
    return network_value(std::make_shared<network_value_truncated_normal_distribution_impl>(
        seed, mean, std_deviation, range));
}

network_value network_value::custom(
    std::function<double(const network_site_info& src, const network_site_info& dest)> func) {
    return network_value(std::make_shared<network_value_custom_impl>(std::move(func)));
}

}  // namespace arb
