#include "../gtest.h"

#include <recipe.hpp>
#include <rss_cell.hpp>
#include <model.hpp>

using namespace nest::mc;

struct boom {};

struct throwing_recipe: public recipe {
    throwing_recipe(unsigned n_cell, unsigned throwing_gid):
        n_(n_cell), boom_(throwing_gid) {}

    cell_size_type num_cells() const override { return n_; }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        return gid==boom_? throw boom{}: rss_cell::rss_cell_description(0, 1, 1000);
    }

    cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::regular_spike_source;
    }

    cell_count_info get_cell_count_info(cell_gid_type) const override {
        return {0, 0, 0};
    }

    std::vector<cell_connection> connections_on(cell_gid_type) const override {
        return std::vector<cell_connection>();
    }

    cell_size_type n_, boom_;
};

void run_model_ctor(const recipe& rec) {
    domain_decomposition dd(rec, group_rules{1u, backend_policy::use_multicore});
    model m(rec, dd);
}

TEST(model_exception, good_recipe) {
    EXPECT_NO_THROW(run_model_ctor(throwing_recipe(10000,-1)));
}

TEST(model_exception, bad_recipe) {
    EXPECT_THROW(run_model_ctor(throwing_recipe(10000,5555)), boom);
}
