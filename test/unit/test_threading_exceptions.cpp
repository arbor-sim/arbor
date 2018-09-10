#include "../gtest.h"

#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/context.hpp>
#include <arbor/mc_cell.hpp>

using namespace arb;

struct throwing_recipe: public recipe {
    throwing_recipe(unsigned n_cell, unsigned throwing_gid):
            n_(n_cell), boom_(throwing_gid) {}

    cell_size_type num_cells() const override { return n_; }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        mc_cell c;
        c.add_soma(6.0);
        return gid==boom_? throw std::exception(): std::move(c);
    }

    cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable1d_neuron;
    }

    std::vector<cell_connection> connections_on(cell_gid_type) const override {
        return std::vector<cell_connection>();
    }

    cell_size_type n_, boom_;
};

void run_model_ctor(const recipe& rec) {

    context ctx = make_context();
    domain_decomposition decomp = partition_load_balance(rec, ctx);
    simulation s(rec, decomp, ctx);
}

TEST(model_exception, good_recipe) {
    EXPECT_NO_THROW(run_model_ctor(throwing_recipe(10000,-1)));
}

TEST(model_exception, bad_recipe) {
    EXPECT_THROW(run_model_ctor(throwing_recipe(10000,5555)), std::exception);
}
