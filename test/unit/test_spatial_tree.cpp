#include <gtest/gtest.h>

#include <arbor/network.hpp>

#include "util/spatial_tree.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <sstream>
#include <vector>

using namespace arb;

namespace {

template <std::size_t DIM>
struct data_point {
    int id = 0;
    std::array<double, DIM> point;

    bool operator<(const data_point<DIM>& p) const {
        return id < p.id || (id == p.id && point < p.point);
    }
};

template <std::size_t DIM>
struct bounding_box_data {
    bounding_box_data(std::size_t seed,
        std::size_t num_points,
        std::array<double, DIM> box_min,
        std::array<double, DIM> box_max):
        box_min(box_min),
        box_max(box_max) {

        std::minstd_rand rand_gen(seed);

        data.reserve(num_points);
        for (std::size_t i = 0; i < num_points; ++i) {
            data_point<DIM> p;
            p.id = i;
            for (std::size_t d = 0; d < DIM; ++d) {

                std::uniform_real_distribution<double> distri(box_min[d], box_max[d]);
                p.point[d] = distri(rand_gen);
            }
            data.emplace_back(p);
        }
    }

    std::array<double, DIM> box_min;
    std::array<double, DIM> box_max;
    std::vector<data_point<DIM>> data;
};

class st_test:
    public ::testing::TestWithParam<
        std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>> {
public:
    void test_spatial_tree() {
        switch (std::get<0>(GetParam())) {
        case 1: test_spatial_tree_dim<1>(); break;
        case 2: test_spatial_tree_dim<2>(); break;
        case 3: test_spatial_tree_dim<3>(); break;
        case 4: test_spatial_tree_dim<4>(); break;
        case 5: test_spatial_tree_dim<5>(); break;
        case 6: test_spatial_tree_dim<6>(); break;
        default: ASSERT_TRUE(false);
        }
    }

private:
    template <std::size_t DIM>
    void test_spatial_tree_dim() {
        std::size_t max_depth = std::get<1>(GetParam());
        std::size_t leaf_size_target = std::get<2>(GetParam());
        std::size_t num_points = std::get<3>(GetParam());

        std::vector<bounding_box_data<DIM>> boxes;
        std::array<double, DIM> box_min, box_max;
        std::vector<data_point<DIM>> data;
        box_min.fill(-10.0);
        box_max.fill(0.0);

        for (std::size_t i = 0; i < DIM; ++i) {
            boxes.emplace_back(1, num_points, box_min, box_max);
            data.insert(data.end(), boxes.back().data.begin(), boxes.back().data.end());
            box_min[i] += 20.0;
            box_max[i] += 20.0;
        }

        spatial_tree<data_point<DIM>, DIM> tree(
            max_depth, leaf_size_target, data, [](const data_point<DIM>& d) { return d.point; });

        // check box without any points
        tree.bounding_box_for_each(
            box_min, box_max, [](const data_point<DIM>& d) { ASSERT_TRUE(false); });

        // check iteration over full tree
        {
            std::vector<data_point<DIM>> tree_data;
            tree.for_each([&](const data_point<DIM>& d) { tree_data.emplace_back(d); });
            ASSERT_EQ(data.size(), tree_data.size());

            std::sort(data.begin(), data.end());
            std::sort(tree_data.begin(), tree_data.end());
            for (std::size_t i = 0; i < data.size(); ++i) {
                ASSERT_EQ(data[i].id, tree_data[i].id);
                ASSERT_EQ(data[i].point, tree_data[i].point);
            }
        }

        // check contents within each box
        for (auto& box: boxes) {
            std::vector<data_point<DIM>> tree_data;
            tree.bounding_box_for_each(box.box_min, box.box_max, [&](const data_point<DIM>& d) {
                tree_data.emplace_back(d);
            });
            ASSERT_EQ(box.data.size(), tree_data.size());

            std::sort(tree_data.begin(), tree_data.end());
            std::sort(box.data.begin(), box.data.end());

            for (std::size_t i = 0; i < box.data.size(); ++i) {
                ASSERT_EQ(box.data[i].id, tree_data[i].id);
                ASSERT_EQ(box.data[i].point, tree_data[i].point);
            }
        }
    }
};

std::string param_type_names(
    const ::testing::TestParamInfo<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>&
        info) {
    std::stringstream stream;

    stream << "dim_" << std::get<0>(info.param);
    stream << "_depth_" << std::get<1>(info.param);
    stream << "_leaf_" << std::get<2>(info.param);
    stream << "_n_" << std::get<3>(info.param);

    return stream.str();
}
}  // namespace

TEST_P(st_test, param) { test_spatial_tree(); }

INSTANTIATE_TEST_SUITE_P(spatial_tree,
    st_test,
    ::testing::Combine(::testing::Values(1, 2, 3),
        ::testing::Values(1, 10, 20),
        ::testing::Values(1, 100, 1000),
        ::testing::Values(0, 1, 10, 100, 1000, 2000)),
    param_type_names);
