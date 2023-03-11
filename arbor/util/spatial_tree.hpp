#pragma once

#include <arbor/common_types.hpp>
#include <arbor/math.hpp>

#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

namespace arb {

// An immutable spatial data structure for storing and iterating over data in "DIM" dimensional
// space. If DIM = 1 it's a binary tree, if DIM = 2 it's a quad tree and so on.
template <typename T, std::size_t DIM>
class spatial_tree {
public:
    static_assert(DIM >= 1, "Dimension of tree must be at least 1.");

    using value_type = T;
    using point_type = std::array<double, DIM>;
    using node_data = std::vector<spatial_tree>;
    using leaf_data = std::vector<T>;
    using location_func_type = point_type (*)(const T &);

    spatial_tree(): size_(0), data_(leaf_data()) {}

    // Create a tree of given maximum depth and target leaf size. If any leaf holds more than the
    // target size, it is recursively split into up to 2^DIM nodes until reaching the maximum depth.
    spatial_tree(std::size_t max_depth,
        std::size_t leaf_size_target,
        leaf_data data,
        location_func_type location):
        size_(data.size()),
        data_(std::move(data)),
        location_(location) {
        auto &leaf_d = std::get<leaf_data>(data_);
        if (leaf_d.empty()) return;

        min_.fill(std::numeric_limits<double>::max());
        max_.fill(-std::numeric_limits<double>::max());

        for (const auto &d: leaf_d) {
            const auto p = location(d);
            for (std::size_t i = 0; i < DIM; ++i) {
                if (p[i] < min_[i]) min_[i] = p[i];
                if (p[i] > max_[i]) max_[i] = p[i];
            }
        }

        point_type mid;
        for (std::size_t i = 0; i < DIM; ++i) { mid[i] = (max_[i] - min_[i]) / 2.0 + min_[i]; }

        if (max_depth > 1 && leaf_d.size() > leaf_size_target) {
            constexpr auto divisor = math::pow<std::size_t, std::size_t>(2, DIM);

            // The initial index of the sub node containing p
            auto sub_node_index = [&](const point_type &p) {
                std::size_t index = 0;
                for (std::size_t i = 0; i < DIM; ++i) { index += i * 2 * (p[i] >= mid[i]); }
                return index;
            };

            node_data new_nodes;
            new_nodes.reserve(divisor);

            // assign each point to sub-node
            std::array<leaf_data, divisor> new_leaf_data;
            for (const auto &d: leaf_d) {
                const auto p = location(d);
                new_leaf_data[sub_node_index(p)].emplace_back(d);
            }

            // move data into new sub-nodes if not empty
            for (auto &l_d: new_leaf_data) {
                if (l_d.size())
                    new_nodes.emplace_back(max_depth - 1, leaf_size_target, std::move(l_d), location);
            }

            // replace current data_ with new sub-nodes
            this->data_ = std::move(new_nodes);
        }
    }

    spatial_tree(const spatial_tree &) = default;

    spatial_tree(spatial_tree &&t) { *this = std::move(t); }

    spatial_tree &operator=(const spatial_tree &) = default;

    spatial_tree &operator=(spatial_tree &&t) {
        data_ = std::move(t.data_);
        size_ = t.size_;
        min_ = t.min_;
        max_ = t.max_;

        t.data_ = leaf_data();
        t.size_ = 0;
        t.min_ = point_type();
        t.max_ = point_type();

        return *this;
    }

    // Iterate over all points recursively.
    // func must have signature `void func(const T&)`.
    template <typename F>
    inline void for_each(const F &func) const {
        std::visit(
            [&](auto &&arg) {
                using arg_type = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<arg_type, node_data>) {
                    for (const auto &node: arg) { node.for_each(func); }
                }
                if constexpr (std::is_same_v<arg_type, leaf_data>) {
                    for (const auto &d: arg) { func(d); }
                }
            },
            data_);
    }

    // Iterate over all points within the given bounding box recursively.
    // func must have signature `void func(const T&)`.
    template <typename F>
    inline void bounding_box_for_each(const point_type &box_min,
        const point_type &box_max,
        const F &func) const {
        auto all_smaller_eq = [](const point_type &lhs, const point_type &rhs) {
            bool result = true;
            for (std::size_t i = 0; i < DIM; ++i) { result &= lhs[i] <= rhs[i]; }
            return result;
        };

        std::visit(
            [&](auto &&arg) {
                using arg_type = std::decay_t<decltype(arg)>;

                if (all_smaller_eq(box_min, min_) && all_smaller_eq(max_, box_max)) {
                    // sub-nodes fully inside box -> call without further boundary
                    // checks
                    if constexpr (std::is_same_v<arg_type, node_data>) {
                        for (const auto &node: arg) { node.template for_each<F>(func); }
                    }
                    if constexpr (std::is_same_v<arg_type, leaf_data>) {
                        for (const auto &d: arg) { func(d); }
                    }
                }
                else {
                    // sub-nodes partially overlap bounding box
                    if constexpr (std::is_same_v<arg_type, node_data>) {
                        for (const auto &node: arg) {
                            if (all_smaller_eq(node.min_, box_max) &&
                                all_smaller_eq(box_min, node.max_))
                                node.template bounding_box_for_each<F>(box_min, box_max, func);
                        }
                    }
                    if constexpr (std::is_same_v<arg_type, leaf_data>) {
                        for (const auto &d: arg) {
                            const auto p = location_(d);
                            if (all_smaller_eq(p, box_max) && all_smaller_eq(box_min, p)) {
                                func(d);
                            }
                        }
                    }
                }
            },
            data_);
    }

    inline std::size_t size() const { return size_; }

    inline bool empty() const { return !size_; }

private:
    std::size_t size_;
    point_type min_, max_;
    std::variant<node_data, leaf_data> data_;
    location_func_type location_;
};

}  // namespace arb
