#include "../gtest.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/version.hpp>

#include "util/span.hpp"

#include "../simple_recipes.hpp"
#include "test.hpp"

#ifdef TEST_MPI
#include <mpi.h>
#endif

using namespace arb;

namespace {
    // Dummy recipes types for testing.

    struct dummy_cell {};
    using homo_recipe = homogeneous_recipe<cell_kind::cable1d_neuron, dummy_cell>;

    // Heterogenous cell population of cable and rss cells.
    // Interleaved so that cells with even gid are cable cells, and even gid are
    // rss cells.
    class hetero_recipe: public recipe {
    public:
        hetero_recipe(cell_size_type s): size_(s)
        {}

        cell_size_type num_cells() const override {
            return size_;
        }

        util::unique_any get_cell_description(cell_gid_type) const override {
            return {};
        }

        cell_kind get_cell_kind(cell_gid_type gid) const override {
            return gid%2?
                cell_kind::spike_source:
                cell_kind::cable1d_neuron;
        }

        cell_size_type num_sources(cell_gid_type) const override { return 0; }
        cell_size_type num_targets(cell_gid_type) const override { return 0; }
        cell_size_type num_probes(cell_gid_type) const override { return 0; }

        std::vector<cell_connection> connections_on(cell_gid_type) const override {
            return {};
        }

        std::vector<event_generator> event_generators(cell_gid_type) const override {
            return {};
        }


    private:
        cell_size_type size_;
    };

    class gj_recipe: public recipe {
    public:
        gj_recipe(unsigned num_ranks): ncopies_(num_ranks){}

        cell_size_type num_cells() const override {
            return size_*ncopies_;
        }

        arb::util::unique_any get_cell_description(cell_gid_type) const override {
            return {};
        }

        cell_kind get_cell_kind(cell_gid_type gid) const override {
            return cell_kind::cable1d_neuron;
        }
        std::vector<cell_gid_type> group_with(cell_gid_type gid) const override {
            unsigned shift = (gid/size_)*size_;
            switch (gid % size_) {
                case 1 :  return {7 + shift};
                case 2 :  return {6 + shift, 9 + shift};
                case 6 :  return {2 + shift, 7 + shift};
                case 7 :  return {6 + shift, 1 + shift};
                case 9 :  return {2 + shift};
                default : return {};
            }
        }

    private:
        cell_size_type size_ = 10;
        unsigned ncopies_;
    };
}

TEST(domain_decomposition, homogeneous_population_mc) {
    // Test on a node with 1 cpu core and no gpus.
    // We assume that all cells will be put into cell groups of size 1.
    // This assumption will not hold in the future, requiring and update to
    // the test.
    proc_allocation resources{1, -1};
#ifdef TEST_MPI
    auto ctx = make_context(resources, MPI_COMM_WORLD);
#else
    auto ctx = make_context(resources);
#endif

    const unsigned N = arb::num_ranks(ctx);
    const unsigned I = arb::rank(ctx);

    // 10 cells per domain
    unsigned n_local = 10;
    unsigned n_global = n_local*N;
    const auto D = partition_load_balance(homo_recipe(n_global, dummy_cell{}), ctx);

    EXPECT_EQ(D.num_global_cells, n_global);
    EXPECT_EQ(D.num_local_cells, n_local);
    EXPECT_EQ(D.groups.size(), n_local);

    auto b = I*n_local;
    auto e = (I+1)*n_local;
    auto gids = util::make_span(b, e);
    for (auto gid: gids) {
        EXPECT_EQ(I, (unsigned)D.gid_domain(gid));
    }

    // Each cell group contains 1 cell of kind cable1d_neuron
    // Each group should also be tagged for cpu execution
    for (auto i: gids) {
        auto local_group = i-b;
        auto& grp = D.groups[local_group];
        EXPECT_EQ(grp.gids.size(), 1u);
        EXPECT_EQ(grp.gids.front(), unsigned(i));
        EXPECT_EQ(grp.backend, backend_kind::multicore);
        EXPECT_EQ(grp.kind, cell_kind::cable1d_neuron);
    }
}

#ifdef ARB_GPU_ENABLED
TEST(domain_decomposition, homogeneous_population_gpu) {
    //  TODO: skip this test
    //      * when the ability to skip tests at runtime is added to Google Test.
    //      * when a GPU is not available
    //  https://github.com/google/googletest/pull/1544

    // Test on a node with 1 gpu and 1 cpu core.
    // Assumes that all cells will be placed on gpu in a single group.

    proc_allocation resources;
    resources.num_threads = 1;
#ifdef TEST_MPI
    auto ctx = make_context(resources, MPI_COMM_WORLD);
#else
    auto ctx = make_context(resources);
#endif

    const unsigned N = arb::num_ranks(ctx);
    const unsigned I = arb::rank(ctx);

    if (!resources.has_gpu()) return; // Skip if no gpu available.

    // 10 cells per domain
    unsigned n_local = 10;
    unsigned n_global = n_local*N;
    const auto D = partition_load_balance(homo_recipe(n_global, dummy_cell{}), ctx);

    EXPECT_EQ(D.num_global_cells, n_global);
    EXPECT_EQ(D.num_local_cells, n_local);
    EXPECT_EQ(D.groups.size(), 1u);

    auto b = I*n_local;
    auto e = (I+1)*n_local;
    auto gids = util::make_span(b, e);
    for (auto gid: gids) {
        EXPECT_EQ(I, (unsigned)D.gid_domain(gid));
    }

    // Each cell group contains 1 cell of kind cable1d_neuron
    // Each group should also be tagged for cpu execution
    auto grp = D.groups[0u];

    EXPECT_EQ(grp.gids.size(), n_local);
    EXPECT_EQ(grp.gids.front(), b);
    EXPECT_EQ(grp.gids.back(), e-1);
    EXPECT_EQ(grp.backend, backend_kind::gpu);
    EXPECT_EQ(grp.kind, cell_kind::cable1d_neuron);
}
#endif

TEST(domain_decomposition, heterogeneous_population) {
    // Test on a node with 1 cpu core and no gpus.
    // We assume that all cells will be put into cell groups of size 1.
    // This assumption will not hold in the future, requiring and update to
    // the test.
    proc_allocation resources{1, -1};
#ifdef TEST_MPI
    auto ctx = make_context(resources, MPI_COMM_WORLD);
#else
    auto ctx = make_context(resources);
#endif

    const auto N = arb::num_ranks(ctx);
    const auto I = arb::rank(ctx);


    // 10 cells per domain
    const unsigned n_local = 10;
    const unsigned n_global = n_local*N;
    const unsigned n_local_grps = n_local; // 1 cell per group
    auto R = hetero_recipe(n_global);
    const auto D = partition_load_balance(R, ctx);

    EXPECT_EQ(D.num_global_cells, n_global);
    EXPECT_EQ(D.num_local_cells, n_local);
    EXPECT_EQ(D.groups.size(), n_local);

    auto b = I*n_local;
    auto e = (I+1)*n_local;
    auto gids = util::make_span(b, e);
    for (auto gid: gids) {
        EXPECT_EQ(I, (unsigned)D.gid_domain(gid));
    }

    // Each cell group contains 1 cell of kind cable1d_neuron
    // Each group should also be tagged for cpu execution
    auto grps = util::make_span(0, n_local_grps);
    std::map<cell_kind, std::set<cell_gid_type>> kind_lists;
    for (auto i: grps) {
        auto& grp = D.groups[i];
        EXPECT_EQ(grp.gids.size(), 1u);
        kind_lists[grp.kind].insert(grp.gids.front());
        EXPECT_EQ(grp.backend, backend_kind::multicore);
    }

    for (auto k: {cell_kind::cable1d_neuron, cell_kind::spike_source}) {
        const auto& gids = kind_lists[k];
        EXPECT_EQ(gids.size(), n_local/2);
        for (auto gid: gids) {
            EXPECT_EQ(k, R.get_cell_kind(gid));
        }
    }
}

TEST(domain_decomposition, compulsory_groups)
{
    proc_allocation resources{1, -1};
    int nranks = 1;
    int rank = 0;
#ifdef TEST_MPI
    auto ctx = make_context(resources, MPI_COMM_WORLD);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    auto ctx = make_context(resources);
#endif
    auto R = gj_recipe(nranks);
    const auto D = partition_load_balance(R, ctx);
    EXPECT_EQ(6u, D.groups.size());

    unsigned shift = rank*nranks;
    std::vector<std::vector<cell_gid_type>> expected_groups =
            { {0 + shift},
              {3 + shift},
              {4 + shift},
              {5 + shift},
              {8 + shift},
              {1 + shift, 2 + shift, 6 + shift, 7 + shift, 9 + shift} };

    for(unsigned i = 0; i < 6u; i++) {
        EXPECT_EQ(expected_groups[i], D.groups[i].gids);
    }
}
