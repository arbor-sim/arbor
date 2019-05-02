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
    using homo_recipe = homogeneous_recipe<cell_kind::cable, dummy_cell>;

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
                cell_kind::cable;
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

    class gj_symmetric: public recipe {
    public:
        gj_symmetric(unsigned num_ranks): ncopies_(num_ranks){}

        cell_size_type num_cells() const override {
            return size_*ncopies_;
        }

        arb::util::unique_any get_cell_description(cell_gid_type) const override {
            return {};
        }

        cell_kind get_cell_kind(cell_gid_type gid) const override {
            return cell_kind::cable;
        }
        std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override {
            unsigned shift = (gid/size_)*size_;
            switch (gid % size_) {
                case 1 :  return { gap_junction_connection({7 + shift, 0}, {gid, 0}, 0.1)};
                case 2 :  return {
                    gap_junction_connection({6 + shift, 0}, {gid, 0}, 0.1),
                    gap_junction_connection({9 + shift, 0}, {gid, 0}, 0.1)
                };
                case 6 :  return {
                    gap_junction_connection({2 + shift, 0}, {gid, 0}, 0.1),
                    gap_junction_connection({7 + shift, 0}, {gid, 0}, 0.1)
                };
                case 7 :  return {
                    gap_junction_connection({6 + shift, 0}, {gid, 0}, 0.1),
                    gap_junction_connection({1 + shift, 0}, {gid, 0}, 0.1)
                };
                case 9 :  return { gap_junction_connection({2 + shift, 0}, {gid, 0}, 0.1)};
                default : return {};
            }
        }

    private:
        cell_size_type size_ = 10;
        unsigned ncopies_;
    };

    class gj_non_symmetric: public recipe {
    public:
        gj_non_symmetric(unsigned num_ranks): groups_(num_ranks), size_(num_ranks){}

        cell_size_type num_cells() const override {
            return size_*groups_;
        }

        arb::util::unique_any get_cell_description(cell_gid_type) const override {
            return {};
        }

        cell_kind get_cell_kind(cell_gid_type gid) const override {
            return cell_kind::cable;
        }
        std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override {
            unsigned group = gid/groups_;
            unsigned id = gid%size_;
            if (id == group && group != (groups_ - 1)) {
                return {gap_junction_connection({gid + size_, 0}, {gid, 0}, 0.1)};
            }
            else if (id == group - 1) {
                return {gap_junction_connection({gid - size_, 0}, {gid, 0}, 0.1)};
            }
            else {
                return {};
            }
        }

    private:
        unsigned groups_;
        cell_size_type size_;
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

    // Each cell group contains 1 cell of kind cable
    // Each group should also be tagged for cpu execution
    for (auto i: gids) {
        auto local_group = i-b;
        auto& grp = D.groups[local_group];
        EXPECT_EQ(grp.gids.size(), 1u);
        EXPECT_EQ(grp.gids.front(), unsigned(i));
        EXPECT_EQ(grp.backend, backend_kind::multicore);
        EXPECT_EQ(grp.kind, cell_kind::cable);
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

    // Each cell group contains 1 cell of kind cable
    // Each group should also be tagged for cpu execution
    auto grp = D.groups[0u];

    EXPECT_EQ(grp.gids.size(), n_local);
    EXPECT_EQ(grp.gids.front(), b);
    EXPECT_EQ(grp.gids.back(), e-1);
    EXPECT_EQ(grp.backend, backend_kind::gpu);
    EXPECT_EQ(grp.kind, cell_kind::cable);
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

    // Each cell group contains 1 cell of kind cable
    // Each group should also be tagged for cpu execution
    auto grps = util::make_span(0, n_local_grps);
    std::map<cell_kind, std::set<cell_gid_type>> kind_lists;
    for (auto i: grps) {
        auto& grp = D.groups[i];
        EXPECT_EQ(grp.gids.size(), 1u);
        kind_lists[grp.kind].insert(grp.gids.front());
        EXPECT_EQ(grp.backend, backend_kind::multicore);
    }

    for (auto k: {cell_kind::cable, cell_kind::spike_source}) {
        const auto& gids = kind_lists[k];
        EXPECT_EQ(gids.size(), n_local/2);
        for (auto gid: gids) {
            EXPECT_EQ(k, R.get_cell_kind(gid));
        }
    }
}

TEST(domain_decomposition, symmetric_groups)
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
    auto R = gj_symmetric(nranks);
    const auto D0 = partition_load_balance(R, ctx);
    EXPECT_EQ(6u, D0.groups.size());

    unsigned shift = rank*R.num_cells()/nranks;
    std::vector<std::vector<cell_gid_type>> expected_groups0 =
            { {0 + shift},
              {3 + shift},
              {4 + shift},
              {5 + shift},
              {8 + shift},
              {1 + shift, 2 + shift, 6 + shift, 7 + shift, 9 + shift}
            };

    for (unsigned i = 0; i < 6; i++){
        EXPECT_EQ(expected_groups0[i], D0.groups[i].gids);
    }

    unsigned cells_per_rank = R.num_cells()/nranks;
    for (unsigned i = 0; i < R.num_cells(); i++) {
        EXPECT_EQ(i/cells_per_rank, (unsigned)D0.gid_domain(i));
    }

    // Test different group_hints
    partition_hint_map hints;
    hints[cell_kind::cable].cpu_group_size = R.num_cells();
    hints[cell_kind::cable].prefer_gpu = false;

    const auto D1 = partition_load_balance(R, ctx, hints);
    EXPECT_EQ(1u, D1.groups.size());

    std::vector<cell_gid_type> expected_groups1 =
            { 0 + shift, 3 + shift, 4 + shift, 5 + shift, 8 + shift,
              1 + shift, 2 + shift, 6 + shift, 7 + shift, 9 + shift };

    EXPECT_EQ(expected_groups1, D1.groups[0].gids);

    for (unsigned i = 0; i < R.num_cells(); i++) {
        EXPECT_EQ(i/cells_per_rank, (unsigned)D1.gid_domain(i));
    }

    hints[cell_kind::cable].cpu_group_size = cells_per_rank/2;
    hints[cell_kind::cable].prefer_gpu = false;

    const auto D2 = partition_load_balance(R, ctx, hints);
    EXPECT_EQ(2u, D2.groups.size());

    std::vector<std::vector<cell_gid_type>> expected_groups2 =
            { { 0 + shift, 3 + shift, 4 + shift, 5 + shift, 8 + shift },
              { 1 + shift, 2 + shift, 6 + shift, 7 + shift, 9 + shift } };

    for (unsigned i = 0; i < 2u; i++) {
        EXPECT_EQ(expected_groups2[i], D2.groups[i].gids);
    }
    for (unsigned i = 0; i < R.num_cells(); i++) {
        EXPECT_EQ(i/cells_per_rank, (unsigned)D2.gid_domain(i));
    }

}

TEST(domain_decomposition, non_symmetric_groups)
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
    /*if (nranks == 1) {
        return;
    }*/

    auto R = gj_non_symmetric(nranks);
    const auto D = partition_load_balance(R, ctx);

    unsigned cells_per_rank = nranks;
    // check groups
    unsigned i = 0;
    for (unsigned gid = rank*cells_per_rank; gid < (rank + 1)*cells_per_rank; gid++) {
        if (gid % nranks == (unsigned)rank - 1) {
            continue;
        }
        else if (gid % nranks == (unsigned)rank && rank != nranks - 1) {
            std::vector<cell_gid_type> cg = {gid, gid+cells_per_rank};
            EXPECT_EQ(cg, D.groups[D.groups.size()-1].gids);
        }
        else {
            std::vector<cell_gid_type> cg = {gid};
            EXPECT_EQ(cg, D.groups[i++].gids);
        }
    }
    // check gid_domains
    for (unsigned gid = 0; gid < R.num_cells(); gid++) {
        auto group = gid/cells_per_rank;
        auto idx = gid % cells_per_rank;
        unsigned ngroups = nranks;
        if (idx == group - 1) {
            EXPECT_EQ(group - 1, (unsigned)D.gid_domain(gid));
        }
        else if (idx == group && group != ngroups - 1) {
            EXPECT_EQ(group, (unsigned)D.gid_domain(gid));
        }
        else {
            EXPECT_EQ(group, (unsigned)D.gid_domain(gid));
        }
    }
}
