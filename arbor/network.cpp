#include <arbor/common_types.hpp>
#include <arbor/network.hpp>

#include <Random123/threefry.h>
#include <Random123/boxmuller.hpp>
#include <Random123/uniform.hpp>

#include <algorithm>
#include <memory>
#include <cmath>
#include <type_traits>
#include <vector>

#include "network_impl.hpp"

namespace arb {

namespace {

// Partial seed to use for network_value and network_selection generation.
// Different seed for each type to avoid unintentional correlation.
enum class network_seed : unsigned {
    selection_bernoulli = 2058443,
    spatial_selection_bernoulli = 839033,
    value_uniform = 48202,
    value_normal = 8405,
    value_truncated_normal = 380237
};

double uniform_rand_from_key_pair(std::array<unsigned, 2> seed,
    network_hash_type key_a,
    network_hash_type key_b) {
    using rand_type = r123::Threefry2x64;
    const rand_type::ctr_type seed_input = {{seed[0], seed[1]}};

    const rand_type::key_type key = {{std::min(key_a, key_b), std::max(key_a, key_b)}};
    rand_type gen;
    return r123::u01<double>(gen(seed_input, key)[0]);
}

double network_location_distance(const network_location& a, const network_location& b) {
    return std::sqrt(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

template <typename Derived>
struct network_selection_crtp: public network_selection_impl {
    bool select_source(cell_gid_type gid,
        const cable_cell& cell,
        const cell_tag_type& tag) const override {
        return static_cast<const Derived*>(this)->select_source_impl(gid, cell, tag);
    }

    bool select_destination(cell_gid_type gid,
        const cable_cell& cell,
        const cell_tag_type& tag) const override {
        return static_cast<const Derived*>(this)->select_destination_impl(gid, cell, tag);
    }

    bool select_source(cell_gid_type gid,
        const lif_cell& cell,
        const cell_tag_type& tag) const override {
        return static_cast<const Derived*>(this)->select_source_impl(gid, cell, tag);
    }

    bool select_destination(cell_gid_type gid,
        const lif_cell& cell,
        const cell_tag_type& tag) const override {
        return static_cast<const Derived*>(this)->select_destination_impl(gid, cell, tag);
    }

    bool select_source(cell_gid_type gid,
        const spike_source_cell& cell,
        const cell_tag_type& tag) const override {
        return static_cast<const Derived*>(this)->select_source_impl(gid, cell, tag);
    }

    bool select_destination(cell_gid_type gid,
        const spike_source_cell& cell,
        const cell_tag_type& tag) const override {
        return static_cast<const Derived*>(this)->select_destination_impl(gid, cell, tag);
    }

    bool select_source(cell_gid_type gid,
        const benchmark_cell& cell,
        const cell_tag_type& tag) const override {
        return static_cast<const Derived*>(this)->select_source_impl(gid, cell, tag);
    }

    bool select_destination(cell_gid_type gid,
        const benchmark_cell& cell,
        const cell_tag_type& tag) const override {
        return static_cast<const Derived*>(this)->select_destination_impl(gid, cell, tag);
    }
};

struct network_selection_all_impl: public network_selection_crtp<network_selection_all_impl> {
    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return true;
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }
};

struct network_selection_none_impl:
    public network_selection_crtp<network_selection_none_impl> {

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return false;
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return false;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return false;
    }
};

struct network_selection_source_cell_kind_impl: public network_selection_impl {
    cell_kind kind;

    explicit network_selection_source_cell_kind_impl(cell_kind k): kind(k) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return src.kind == kind;
    }

    bool select_source(cell_gid_type gid,
        const cable_cell& cell,
        const cell_tag_type& tag) const override {
        return kind == cell_kind::cable;
    }

    bool select_source(cell_gid_type gid,
        const lif_cell& cell,
        const cell_tag_type& tag) const override {
        return kind == cell_kind::lif;
    }

    bool select_source(cell_gid_type gid,
        const benchmark_cell& cell,
        const cell_tag_type& tag) const override {
        return kind == cell_kind::benchmark;
    }

    bool select_source(cell_gid_type gid,
        const spike_source_cell& cell,
        const cell_tag_type& tag) const override {
        return kind == cell_kind::spike_source;
    }

    bool select_destination(cell_gid_type gid,
        const cable_cell& cell,
        const cell_tag_type& tag) const override {
        return true;
    }

    bool select_destination(cell_gid_type gid,
        const lif_cell& cell,
        const cell_tag_type& tag) const override {
        return true;
    }

    bool select_destination(cell_gid_type gid,
        const benchmark_cell& cell,
        const cell_tag_type& tag) const override {
        return true;
    }

    bool select_destination(cell_gid_type gid,
        const spike_source_cell& cell,
        const cell_tag_type& tag) const override {
        return true;
    }
};

struct network_selection_destination_cell_kind_impl:
    public network_selection_impl {
    cell_kind kind;

    explicit network_selection_destination_cell_kind_impl(cell_kind k): kind(k) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return src.kind == kind;
    }

    bool select_source(cell_gid_type gid,
        const cable_cell& cell,
        const cell_tag_type& tag) const override {
        return true;
    }

    bool select_source(cell_gid_type gid,
        const lif_cell& cell,
        const cell_tag_type& tag) const override {
        return true;
    }

    bool select_source(cell_gid_type gid,
        const benchmark_cell& cell,
        const cell_tag_type& tag) const override {
        return true;
    }

    bool select_source(cell_gid_type gid,
        const spike_source_cell& cell,
        const cell_tag_type& tag) const override {
        return true;
    }

    bool select_destination(cell_gid_type gid,
        const cable_cell& cell,
        const cell_tag_type& tag) const override {
        return kind == cell_kind::cable;
    }

    bool select_destination(cell_gid_type gid,
        const lif_cell& cell,
        const cell_tag_type& tag) const override {
        return kind == cell_kind::lif;
    }

    bool select_destination(cell_gid_type gid,
        const benchmark_cell& cell,
        const cell_tag_type& tag) const override {
        return kind == cell_kind::benchmark;
    }

    bool select_destination(cell_gid_type gid,
        const spike_source_cell& cell,
        const cell_tag_type& tag) const override {
        return kind == cell_kind::spike_source;
    }
};


struct network_selection_source_label_impl:
    public network_selection_crtp<network_selection_source_label_impl> {
    std::vector<cell_tag_type> sorted_labels;

    explicit network_selection_source_label_impl(std::vector<cell_tag_type> labels):
        sorted_labels(std::move(labels)) {
        std::sort(sorted_labels.begin(), sorted_labels.end());
    }

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return std::binary_search(sorted_labels.begin(), sorted_labels.end(), src.tag);
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return std::binary_search(sorted_labels.begin(), sorted_labels.end(), tag);
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }
};

struct network_selection_destination_label_impl:
    public network_selection_crtp<network_selection_destination_label_impl> {
    std::vector<cell_tag_type> sorted_labels;

    explicit network_selection_destination_label_impl(std::vector<cell_tag_type> labels):
        sorted_labels(std::move(labels)) {
        std::sort(sorted_labels.begin(), sorted_labels.end());
    }

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return std::binary_search(sorted_labels.begin(), sorted_labels.end(), dest.tag);
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return std::binary_search(sorted_labels.begin(), sorted_labels.end(), tag);
    }
};

struct network_selection_source_gid_impl:
    public network_selection_crtp<network_selection_source_gid_impl> {
    std::vector<cell_gid_type> sorted_gids;

    explicit network_selection_source_gid_impl(std::vector<cell_gid_type> gids):
        sorted_gids(std::move(gids)) {
        std::sort(sorted_gids.begin(), sorted_gids.end());
    }

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return std::binary_search(sorted_gids.begin(), sorted_gids.end(), src.gid);
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return std::binary_search(sorted_gids.begin(), sorted_gids.end(), gid);
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }
};

struct network_selection_destination_gid_impl:
    public network_selection_crtp<network_selection_destination_gid_impl> {
    std::vector<cell_gid_type> sorted_gids;

    explicit network_selection_destination_gid_impl(std::vector<cell_gid_type> gids):
        sorted_gids(std::move(gids)) {
        std::sort(sorted_gids.begin(), sorted_gids.end());
    }

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return std::binary_search(sorted_gids.begin(), sorted_gids.end(), dest.gid);
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return std::binary_search(sorted_gids.begin(), sorted_gids.end(), gid);
    }
};


struct network_selection_invert_impl:
    public network_selection_crtp<network_selection_invert_impl> {
    std::shared_ptr<network_selection_impl> selection;

    explicit network_selection_invert_impl(std::shared_ptr<network_selection_impl> s):
        selection(std::move(s)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return !selection->select_connection(src, dest);
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;  // cannot exclude any because source selection cannot be inverted without
                      // knowing selection criteria.
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;  // cannot exclude any because destination selection cannot be inverted without
                      // knowing selection criteria.
    }
};


struct network_selection_inter_cell_impl: public network_selection_crtp<network_selection_inter_cell_impl> {
    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const {
        return src.gid != dest.gid;
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }
};

struct network_selection_not_equal_impl: public network_selection_crtp<network_selection_not_equal_impl> {
    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const {
        return src.gid != dest.gid || src.tag != dest.tag || src.location != dest.location;
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }
};

struct network_selection_custom_impl:
    public network_selection_crtp<network_selection_custom_impl> {
        std::function<bool(const network_site_info& src, const network_site_info& dest)> func;

        explicit network_selection_custom_impl(
            std::function<bool(const network_site_info& src, const network_site_info& dest)> f):
            func(std::move(f)) {}

        bool select_connection(const network_site_info& src,
            const network_site_info& dest) const override {
        return func(src, dest);
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }
};

struct network_selection_within_distance_impl:
    public network_selection_crtp<network_selection_within_distance_impl> {
    double distance;

    explicit network_selection_within_distance_impl(double distance): distance(distance) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return network_location_distance(src.location, dest.location) <= distance;
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    std::optional<double> max_distance() const override { return distance; }
};


struct network_selection_bernoulli_random_impl:
    public network_selection_crtp<network_selection_bernoulli_random_impl> {
    unsigned seed;
    double probability;

    network_selection_bernoulli_random_impl(unsigned seed, double p): seed(seed), probability(p) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return uniform_rand_from_key_pair({unsigned(network_seed::selection_bernoulli), seed},
                   src.hash,
                   dest.hash) < probability;
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }
};


struct network_selection_linear_bernoulli_random_impl:
    public network_selection_crtp<network_selection_linear_bernoulli_random_impl> {
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
        const double distance = network_location_distance(src.location, dest.location);

        if(distance < distance_begin || distance > distance_end) return false;

        const double p =
            (p_begin * (distance_end - distance) + p_end * (distance - distance_begin)) /
            (distance_end - distance_begin);

        return uniform_rand_from_key_pair(
                   {unsigned(network_seed::spatial_selection_bernoulli), seed},
                   src.hash,
                   dest.hash) < p;
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    std::optional<double> max_distance() const override { return distance_end; }
};

struct network_selection_and_impl: public network_selection_crtp<network_selection_and_impl> {
    std::shared_ptr<network_selection_impl> left, right;

    network_selection_and_impl(std::shared_ptr<network_selection_impl> l,
        std::shared_ptr<network_selection_impl> r):
        left(std::move(l)),
        right(std::move(r)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return left->select_connection(src, dest) && right->select_connection(src, dest);
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return left->select_source(gid, cell, tag) && right->select_source(gid, cell, tag);
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return left->select_destination(gid, cell, tag) &&
               right->select_destination(gid, cell, tag);
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

struct network_selection_or_impl: public network_selection_crtp<network_selection_or_impl> {
    std::shared_ptr<network_selection_impl> left, right;

    network_selection_or_impl(std::shared_ptr<network_selection_impl> l,
        std::shared_ptr<network_selection_impl> r):
        left(std::move(l)),
        right(std::move(r)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return left->select_connection(src, dest) || right->select_connection(src, dest);
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return left->select_source(gid, cell, tag) || right->select_source(gid, cell, tag);
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return left->select_destination(gid, cell, tag) ||
               right->select_destination(gid, cell, tag);
    }

    std::optional<double> max_distance() const override {
        const auto d_left = left->max_distance();
        const auto d_right = right->max_distance();

        if (d_left && d_right) return std::max(d_left.value(), d_right.value());

        return std::nullopt;
    }
};

struct network_selection_xor_impl: public network_selection_crtp<network_selection_xor_impl> {
    std::shared_ptr<network_selection_impl> left, right;

    network_selection_xor_impl(std::shared_ptr<network_selection_impl> l,
        std::shared_ptr<network_selection_impl> r):
        left(std::move(l)),
        right(std::move(r)) {}

    bool select_connection(const network_site_info& src,
        const network_site_info& dest) const override {
        return left->select_connection(src, dest) ^ right->select_connection(src, dest);
    }

    template <typename CellType>
    bool select_source_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    template <typename CellType>
    bool select_destination_impl(cell_gid_type gid,
        const CellType& cell,
        const cell_tag_type& tag) const {
        return true;
    }

    std::optional<double> max_distance() const override {
        const auto d_left = left->max_distance();
        const auto d_right = right->max_distance();

        if (d_left && d_right) return std::max(d_left.value(), d_right.value());

        return std::nullopt;
    }
};

}

}  // namespace arb
