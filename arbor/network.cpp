#include <arbor/common_types.hpp>
#include <arbor/network.hpp>

#include <Random123/threefry.h>
#include <Random123/boxmuller.hpp>
#include <Random123/uniform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>
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

double network_location_distance(const mpoint& a, const mpoint& b) {
    return std::sqrt(a.x * b.x + a.y * b.y + a.z * b.z);
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

struct network_selection_complement_impl: public network_selection_impl {
    std::shared_ptr<network_selection_impl> selection;

    explicit network_selection_complement_impl(std::shared_ptr<network_selection_impl> s):
        selection(std::move(s)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return !selection->select_connection(src, dest);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;  // cannot exclude any because source selection cannot be complemented without
                      // knowing selection criteria.
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return true;  // cannot exclude any because destination selection cannot be complemented
                      // without knowing selection criteria.
    }

    void initialize(const network_label_dict& dict) override { selection->initialize(dict); };
};

struct network_selection_named_impl: public network_selection_impl {
    using impl_pointer_type = std::shared_ptr<network_selection_impl>;

    std::variant<impl_pointer_type, std::string> selection;

    explicit network_selection_named_impl(std::string name): selection(std::move(name)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        if (!std::holds_alternative<impl_pointer_type>(selection))
            throw arbor_internal_error("Trying to use unitialized named network selection.");
        return std::get<impl_pointer_type>(selection)->select_connection(src, dest);
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        if (!std::holds_alternative<impl_pointer_type>(selection))
            throw arbor_internal_error("Trying to use unitialized named network selection.");
        return std::get<impl_pointer_type>(selection)->select_source(kind, gid, label);
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        if (!std::holds_alternative<impl_pointer_type>(selection))
            throw arbor_internal_error("Trying to use unitialized named network selection.");
        return std::get<impl_pointer_type>(selection)->select_destination(kind, gid, label);
    }

    void initialize(const network_label_dict& dict) override {
        if (std::holds_alternative<std::string>(selection)) {
            auto s = dict.selection(std::get<std::string>(selection));
            if (!s.has_value())
                throw arbor_exception(std::string("Network selection with label \"") +
                                      std::get<std::string>(selection) + "\" not found.");
            selection = thingify(s.value(), dict);
        }
    };
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

struct network_selection_custom_impl: public network_selection_impl {
    network_selection::custom_func_type func;

    explicit network_selection_custom_impl(network_selection::custom_func_type f):
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

struct network_selection_distance_lt_impl: public network_selection_impl {
    double distance;

    explicit network_selection_distance_lt_impl(double distance): distance(distance) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return network_location_distance(src.global_location, dest.global_location) < distance;
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

struct network_selection_distance_gt_impl: public network_selection_impl {
    double distance;

    explicit network_selection_distance_gt_impl(double distance): distance(distance) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return network_location_distance(src.global_location, dest.global_location) > distance;
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

struct network_selection_random_bernoulli_impl: public network_selection_impl {
    unsigned seed;
    double probability;

    network_selection_random_bernoulli_impl(unsigned seed, double p): seed(seed), probability(p) {}

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

struct network_selection_random_linear_distance_impl: public network_selection_impl {
    unsigned seed;
    double distance_begin;
    double p_begin;
    double distance_end;
    double p_end;

    network_selection_random_linear_distance_impl(unsigned seed_,
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
        const double distance =
            network_location_distance(src.global_location, dest.global_location);

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

struct network_selection_intersect_impl: public network_selection_impl {
    std::shared_ptr<network_selection_impl> left, right;

    network_selection_intersect_impl(std::shared_ptr<network_selection_impl> l,
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

    void initialize(const network_label_dict& dict) override {
        left->initialize(dict);
        right->initialize(dict);
    };
};

struct network_selection_join_impl: public network_selection_impl {
    std::shared_ptr<network_selection_impl> left, right;

    network_selection_join_impl(std::shared_ptr<network_selection_impl> l,
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

    void initialize(const network_label_dict& dict) override {
        left->initialize(dict);
        right->initialize(dict);
    };
};

struct network_selection_symmetric_difference_impl: public network_selection_impl {
    std::shared_ptr<network_selection_impl> left, right;

    network_selection_symmetric_difference_impl(std::shared_ptr<network_selection_impl> l,
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

    void initialize(const network_label_dict& dict) override {
        left->initialize(dict);
        right->initialize(dict);
    };
};

struct network_selection_difference_impl: public network_selection_impl {
    std::shared_ptr<network_selection_impl> left, right;

    network_selection_difference_impl(std::shared_ptr<network_selection_impl> l,
        std::shared_ptr<network_selection_impl> r):
        left(std::move(l)),
        right(std::move(r)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return left->select_connection(src, dest) && !(right->select_connection(src, dest));
    }

    bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return left->select_source(kind, gid, label);
    }

    bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& label) const override {
        return left->select_destination(kind, gid, label);
    }

    std::optional<double> max_distance() const override {
        const auto d_left = left->max_distance();

        if (d_left) return d_left.value();

        return std::nullopt;
    }

    void initialize(const network_label_dict& dict) override {
        left->initialize(dict);
        right->initialize(dict);
    };
};

struct network_value_scalar_impl: public network_value_impl {
    double value;

    network_value_scalar_impl(double v): value(v) {}

    double get(const network_site_info& src, const network_site_info& dest) const override {
        return value;
    }
};

struct network_value_uniform_distribution_impl: public network_value_impl {
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
    network_value::custom_func_type func;

    network_value_custom_impl(network_value::custom_func_type f): func(std::move(f)) {}

    double get(const network_site_info& src, const network_site_info& dest) const override {
        return func(src, dest);
    }
};

struct network_value_named_impl: public network_value_impl {
    using impl_pointer_type = std::shared_ptr<network_value_impl>;

    std::variant<impl_pointer_type, std::string> value;

    explicit network_value_named_impl(std::string name): value(std::move(name)) {}

    double get(const network_site_info& src, const network_site_info& dest) const override {
        if (!std::holds_alternative<impl_pointer_type>(value))
            throw arbor_internal_error("Trying to use unitialized named network value.");
        return std::get<impl_pointer_type>(value)->get(src, dest);
    }

    void initialize(const network_label_dict& dict) override {
        if (std::holds_alternative<std::string>(value)) {
            auto s = dict.value(std::get<std::string>(value));
            if (!s.has_value())
                throw arbor_exception(std::string("Network value with label \"") +
                                      std::get<std::string>(value) + "\" not found.");
            value = thingify(s.value(), dict);
        }
    };
};

}  // namespace

network_site_info::network_site_info(cell_gid_type gid,
    cell_lid_type lid,
    cell_kind kind,
    std::string_view label,
    mlocation location,
    mpoint global_location):
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

network_selection network_selection::intersect(network_selection left, network_selection right) {
    return network_selection(std::make_shared<network_selection_intersect_impl>(
        std::move(left.impl_), std::move(right.impl_)));
}

network_selection network_selection::join(network_selection left, network_selection right) {
    return network_selection(std::make_shared<network_selection_join_impl>(
        std::move(left.impl_), std::move(right.impl_)));
}

network_selection network_selection::symmetric_difference(network_selection left,
    network_selection right) {
    return network_selection(std::make_shared<network_selection_symmetric_difference_impl>(
        std::move(left.impl_), std::move(right.impl_)));
}

network_selection network_selection::difference(network_selection left, network_selection right) {
    return network_selection(std::make_shared<network_selection_difference_impl>(
        std::move(left.impl_), std::move(right.impl_)));
}

network_selection network_selection::all() {
    return network_selection(std::make_shared<network_selection_all_impl>());
}

network_selection network_selection::none() {
    return network_selection(std::make_shared<network_selection_none_impl>());
}

network_selection network_selection::named(std::string name) {
    return network_selection(std::make_shared<network_selection_named_impl>(std::move(name)));
}

network_selection network_selection::source_cell_kind(cell_kind kind) {
    return network_selection(std::make_shared<network_selection_source_cell_kind_impl>(kind));
}

network_selection network_selection::destination_cell_kind(cell_kind kind) {
    return network_selection(std::make_shared<network_selection_destination_cell_kind_impl>(kind));
}

network_selection network_selection::source_label(std::vector<cell_tag_type> labels) {
    return network_selection(
        std::make_shared<network_selection_source_label_impl>(std::move(labels)));
}

network_selection network_selection::destination_label(std::vector<cell_tag_type> labels) {
    return network_selection(
        std::make_shared<network_selection_destination_label_impl>(std::move(labels)));
}

network_selection network_selection::source_gid(std::vector<cell_gid_type> gids) {
    return network_selection(std::make_shared<network_selection_source_gid_impl>(std::move(gids)));
}

network_selection network_selection::destination_gid(std::vector<cell_gid_type> gids) {
    return network_selection(
        std::make_shared<network_selection_destination_gid_impl>(std::move(gids)));
}

network_selection network_selection::complement(network_selection s) {
    return network_selection(
        std::make_shared<network_selection_complement_impl>(std::move(s.impl_)));
}

network_selection network_selection::inter_cell() {
    return network_selection(std::make_shared<network_selection_inter_cell_impl>());
}

network_selection network_selection::random_bernoulli(unsigned seed, double p) {
    return network_selection(std::make_shared<network_selection_random_bernoulli_impl>(seed, p));
}

network_selection network_selection::custom(custom_func_type func) {
    return network_selection(std::make_shared<network_selection_custom_impl>(std::move(func)));
}

network_selection network_selection::distance_lt(double distance) {
    return network_selection(std::make_shared<network_selection_distance_lt_impl>(distance));
}

network_selection network_selection::distance_gt(double distance) {
    return network_selection(std::make_shared<network_selection_distance_gt_impl>(distance));
}

network_selection network_selection::random_linear_distance(unsigned seed,
    double distance_begin,
    double p_begin,
    double distance_end,
    double p_end) {
    return network_selection(std::make_shared<network_selection_random_linear_distance_impl>(
        seed, distance_begin, p_begin, distance_end, p_end));
}

network_value::network_value(std::shared_ptr<network_value_impl> impl): impl_(std::move(impl)) {}

network_value network_value::scalar(double value) {
    return network_value(std::make_shared<network_value_scalar_impl>(value));
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

network_value network_value::custom(custom_func_type func) {
    return network_value(std::make_shared<network_value_custom_impl>(std::move(func)));
}

network_value network_value::named(std::string name) {
    return network_value(std::make_shared<network_value_named_impl>(std::move(name)));
}

network_label_dict& network_label_dict::set(const std::string& name, network_selection s) {
    selections_.insert_or_assign(name, std::move(s));
    return *this;
}

network_label_dict& network_label_dict::set(const std::string& name, network_value v) {
    values_.insert_or_assign(name, std::move(v));
    return *this;
}

std::optional<network_selection> network_label_dict::selection(const std::string& name) const {
    auto it = selections_.find(name);
    if (it != selections_.end()) return it->second;

    return std::nullopt;
}

std::optional<network_value> network_label_dict::value(const std::string& name) const {
    auto it = values_.find(name);
    if (it != values_.end()) return it->second;

    return std::nullopt;
}

}  // namespace arb
