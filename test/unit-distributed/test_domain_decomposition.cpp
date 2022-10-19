#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arbor/context.hpp>
#include <arbor/domdecexcept.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/version.hpp>

#include <arborenv/default_env.hpp>

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
        gj_symmetric(unsigned num_ranks, bool fully_connected):
            ncopies_(num_ranks),
            fully_connected_(fully_connected) {}

        cell_size_type num_cells_per_rank() const {
            return size_;
        }

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
                case 1 :  {
                    if (!fully_connected_) return {};
                    return {gap_junction_connection({7 + shift, "gj"}, {"gj"}, 0.1)};
                }
                case 2 :  {
                    if (!fully_connected_) return {};
                    return {
                        gap_junction_connection({6 + shift, "gj"}, {"gj"}, 0.1),
                        gap_junction_connection({9 + shift, "gj"}, {"gj"}, 0.1)
                    };
                }
                case 6 :  return {
                    gap_junction_connection({2 + shift, "gj"}, {"gj"}, 0.1),
                    gap_junction_connection({7 + shift, "gj"}, {"gj"}, 0.1)
                };
                case 7 :  {
                    if (!fully_connected_)  {
                        return {gap_junction_connection({1 + shift, "gj"}, {"gj"}, 0.1)};
                    }
                    return {
                        gap_junction_connection({6 + shift, "gj"}, {"gj"}, 0.1),
                        gap_junction_connection({1 + shift, "gj"}, {"gj"}, 0.1)
                    };
                }
                case 9 :  return { gap_junction_connection({2 + shift, "gj"}, {"gj"}, 0.1)};
                default : return {};
            }
        }

    private:
        cell_size_type size_ = 10;
        unsigned ncopies_;
        bool fully_connected_;
    };

    class gj_multi_group: public recipe {
    public:
        gj_multi_group(unsigned num_ranks, bool fully_connected):
            groups_(num_ranks),
            size_(num_ranks),
            fully_connected_(fully_connected) {}

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
            // Example topology on 4 ranks of 4 cells each
            // Fully connected             Not fully connected
            // 0  1  2  3                  0  1  2  3
            // ^                           |
            // |                           |
            // v                           v
            // 4  5  6  7                  4  5  6  7
            //    ^                           |
            //    |                           |
            //    v                           v
            // 8  9  10 11                 8  9  10 11
            //       ^                           |
            //       |                           |
            //       v                           v
            // 12 13 14 15                 12 13 14 15
            unsigned group = gid/groups_;
            unsigned id = gid%size_;
            if (id == group && group != (groups_ - 1) && fully_connected_) {
                return {gap_junction_connection({gid + size_, "gj"}, {"gj"}, 0.1)};
            }
            else if (id == group - 1) {
                return {gap_junction_connection({gid - size_, "gj"}, {"gj"}, 0.1)};
            }
            else {
                return {};
            }
        }

    private:
        unsigned groups_;
        cell_size_type size_;
        bool fully_connected_;
    };

    class gj_single_group: public recipe {
    public:
        gj_single_group(unsigned num_ranks):
            groups_(num_ranks),
            size_(num_ranks){}

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
            // Example topology on 4 ranks of 4 cells each
            // 0  1  2  3
            //    |
            //    v
            // 4  5  6  7
            //    |
            //    v
            // 8  9  10 11
            //    |
            //    v
            // 12 13 14 15
            unsigned group = gid/groups_;
            unsigned id = gid%size_;
            if (group!= 0 && id == 1) {
                return {gap_junction_connection({gid - size_, "gj"}, {"gj"}, 0.1)};
            }
            return {};
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

    auto rec = homo_recipe(n_global, dummy_cell{});
    const auto D = partition_load_balance(rec, ctx);

    EXPECT_EQ(D.num_global_cells(), (unsigned)n_global);
    EXPECT_EQ(D.num_local_cells(), n_local);
    EXPECT_EQ(D.num_groups(), n_local);

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
        auto& grp = D.group(local_group);
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
    auto rec = homo_recipe(n_global, dummy_cell{});
    const auto D = partition_load_balance(rec, ctx);

    EXPECT_EQ(D.num_global_cells(), n_global);
    EXPECT_EQ(D.num_local_cells(), n_local);
    EXPECT_EQ(D.num_groups(), 1u);

    auto b = I*n_local;
    auto e = (I+1)*n_local;
    auto gids = util::make_span(b, e);
    for (auto gid: gids) {
        EXPECT_EQ(I, (unsigned)D.gid_domain(gid));
    }

    // Each cell group contains 1 cell of kind cable
    // Each group should also be tagged for cpu execution
    auto grp = D.group(0u);

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

    EXPECT_EQ(D.num_global_cells(), n_global);
    EXPECT_EQ(D.num_local_cells(), n_local);
    EXPECT_EQ(D.num_groups(), n_local);

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
        auto& grp = D.group(i);
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

TEST(domain_decomposition, symmetric_groups) {
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
    std::vector<gj_symmetric> recipes = {gj_symmetric(nranks, true), gj_symmetric(nranks, false)};
    for (const auto& R: recipes) {
        const auto D0 = partition_load_balance(R, ctx);
        EXPECT_EQ(6u, D0.num_groups());

        unsigned shift = rank * R.num_cells()/nranks;
        std::vector<std::vector<cell_gid_type>> expected_groups0 =
                {{0 + shift},
                 {3 + shift},
                 {4 + shift},
                 {5 + shift},
                 {8 + shift},
                 {1 + shift, 2 + shift, 6 + shift, 7 + shift, 9 + shift}
                };

        for (unsigned i = 0; i < 6; i++) {
            EXPECT_EQ(expected_groups0[i], D0.group(i).gids);
        }

        unsigned cells_per_rank = R.num_cells()/nranks;
        for (unsigned i = 0; i < R.num_cells(); i++) {
            EXPECT_EQ(i/cells_per_rank, (unsigned) D0.gid_domain(i));
        }

        // Test different group_hints
        partition_hint_map hints;
        hints[cell_kind::cable].cpu_group_size = R.num_cells();
        hints[cell_kind::cable].prefer_gpu = false;

        const auto D1 = partition_load_balance(R, ctx, hints);
        EXPECT_EQ(1u, D1.num_groups());

        std::vector<cell_gid_type> expected_groups1 =
                {0 + shift, 3 + shift, 4 + shift, 5 + shift, 8 + shift,
                 1 + shift, 2 + shift, 6 + shift, 7 + shift, 9 + shift};

        EXPECT_EQ(expected_groups1, D1.group(0).gids);

        for (unsigned i = 0; i < R.num_cells(); i++) {
            EXPECT_EQ(i/cells_per_rank, (unsigned) D1.gid_domain(i));
        }

        hints[cell_kind::cable].cpu_group_size = cells_per_rank/2;
        hints[cell_kind::cable].prefer_gpu = false;

        const auto D2 = partition_load_balance(R, ctx, hints);
        EXPECT_EQ(2u, D2.num_groups());

        std::vector<std::vector<cell_gid_type>> expected_groups2 =
                {{0 + shift, 3 + shift, 4 + shift, 5 + shift, 8 + shift},
                 {1 + shift, 2 + shift, 6 + shift, 7 + shift, 9 + shift}};

        for (unsigned i = 0; i < 2u; i++) {
            EXPECT_EQ(expected_groups2[i], D2.group(i).gids);
        }
        for (unsigned i = 0; i < R.num_cells(); i++) {
            EXPECT_EQ(i/cells_per_rank, (unsigned) D2.gid_domain(i));
        }
    }
}

TEST(domain_decomposition, gj_multi_distributed_groups) {
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
    std::vector<gj_multi_group> recipes = {gj_multi_group(nranks, true), gj_multi_group(nranks, false)};
    for (const auto& R: recipes) {
        const auto D = partition_load_balance(R, ctx);

        unsigned cells_per_rank = nranks;
        // check groups
        unsigned i = 0;
        for (unsigned gid = rank * cells_per_rank; gid < (rank + 1) * cells_per_rank; gid++) {
            if (gid % nranks == (unsigned) rank - 1) {
                continue;
            } else if (gid % nranks == (unsigned) rank && rank != nranks - 1) {
                std::vector<cell_gid_type> cg = {gid, gid + cells_per_rank};
                EXPECT_EQ(cg, D.group(D.num_groups() - 1).gids);
            } else {
                std::vector<cell_gid_type> cg = {gid};
                EXPECT_EQ(cg, D.group(i++).gids);
            }
        }
        // check gid_domains
        for (unsigned gid = 0; gid < R.num_cells(); gid++) {
            auto group = gid / cells_per_rank;
            auto idx = gid % cells_per_rank;
            unsigned ngroups = nranks;
            if (idx == group - 1) {
                EXPECT_EQ(group - 1, (unsigned) D.gid_domain(gid));
            } else if (idx == group && group != ngroups - 1) {
                EXPECT_EQ(group, (unsigned) D.gid_domain(gid));
            } else {
                EXPECT_EQ(group, (unsigned) D.gid_domain(gid));
            }
        }
    }
}

TEST(domain_decomposition, gj_single_distributed_group) {
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
    auto R = gj_single_group(nranks);
    const auto D = partition_load_balance(R, ctx);

    unsigned cells_per_rank = nranks;
    // check groups
    unsigned i = 0;
    for (unsigned gid = rank*cells_per_rank; gid < (rank + 1)*cells_per_rank; gid++) {
        if (gid%nranks == 1) {
            if (rank == 0) {
                std::vector<cell_gid_type> cg;
                for (int r = 0; r < nranks; ++r) {
                    cg.push_back(gid + (r*nranks));
                }
                EXPECT_EQ(cg, D.groups().back().gids);
            } else {
                continue;
            }
        } else {
            std::vector<cell_gid_type> cg = {gid};
            EXPECT_EQ(cg, D.group(i++).gids);
        }
    }
    // check gid_domains
    for (unsigned gid = 0; gid < R.num_cells(); gid++) {
        auto group = gid/cells_per_rank;
        auto idx = gid%cells_per_rank;
        if (idx == 1) {
            EXPECT_EQ(0u, (unsigned) D.gid_domain(gid));
        } else {
            EXPECT_EQ(group, (unsigned) D.gid_domain(gid));
        }
    }
}

TEST(domain_decomposition, partition_by_group)
{
    proc_allocation resources{1,  arbenv::default_gpu()};
    int nranks = 1;
    int rank = 0;
#ifdef TEST_MPI
    auto ctx = make_context(resources, MPI_COMM_WORLD);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    auto ctx = make_context(resources);
#endif

    {
        const unsigned cells_per_rank = 10;
        auto rec = homo_recipe(cells_per_rank*nranks, dummy_cell{});
        std::vector<cell_gid_type> gids;
        for (unsigned i = 0; i < cells_per_rank; ++i) {
            gids.push_back(rank*cells_per_rank + i);
        }
#ifdef ARB_GPU_ENABLED
        auto d = domain_decomposition(rec, ctx, {{cell_kind::cable, gids, backend_kind::gpu}});
#else
        auto d = domain_decomposition(rec, ctx, {{cell_kind::cable, gids, backend_kind::multicore}});
#endif

        EXPECT_EQ(nranks, d.num_domains());
        EXPECT_EQ(rank, d.domain_id());
        EXPECT_EQ(cells_per_rank, d.num_local_cells());
        EXPECT_EQ(cells_per_rank*nranks, d.num_global_cells());
        EXPECT_EQ(1u, d.num_groups());
        EXPECT_EQ(gids, d.group(0).gids);
        EXPECT_EQ(cell_kind::cable, d.group(0).kind);
#ifdef ARB_GPU_ENABLED
        EXPECT_EQ(backend_kind::gpu, d.group(0).backend);
#else
        EXPECT_EQ(backend_kind::multicore, d.group(0).backend);
#endif
        for (unsigned i = 0; i < cells_per_rank*nranks; ++i) {
            EXPECT_EQ((int)(i/cells_per_rank), d.gid_domain(i));
        }
    }
    {
        auto rec = gj_symmetric(nranks, true);
        const unsigned cells_per_rank = rec.num_cells_per_rank();
        const unsigned shift = cells_per_rank*rank;

        std::vector<cell_gid_type> gids0 = {1+shift, 2+shift, 6+shift, 7+shift, 9+shift};
        std::vector<cell_gid_type> gids1 = {3+shift};
        std::vector<cell_gid_type> gids2 = {0+shift, 4+shift, 5+shift, 8+shift};

        group_description g0 = {cell_kind::lif,   gids0, backend_kind::multicore};
        group_description g1 = {cell_kind::cable, gids1, backend_kind::multicore};
#ifdef ARB_GPU_ENABLED
        group_description g2 = {cell_kind::cable, gids2, backend_kind::gpu};
#else
        group_description g2 = {cell_kind::cable, gids2, backend_kind::multicore};
#endif
        auto d = domain_decomposition(rec, ctx, {g0, g1, g2});

        EXPECT_EQ(nranks, d.num_domains());
        EXPECT_EQ(rank, d.domain_id());
        EXPECT_EQ(cells_per_rank, d.num_local_cells());
        EXPECT_EQ(cells_per_rank*nranks, d.num_global_cells());
        EXPECT_EQ(3u, d.num_groups());

        EXPECT_EQ(gids0, d.group(0).gids);
        EXPECT_EQ(gids1, d.group(1).gids);
        EXPECT_EQ(gids2, d.group(2).gids);

        EXPECT_EQ(cell_kind::lif,   d.group(0).kind);
        EXPECT_EQ(cell_kind::cable, d.group(1).kind);
        EXPECT_EQ(cell_kind::cable, d.group(2).kind);

        EXPECT_EQ(backend_kind::multicore, d.group(0).backend);
        EXPECT_EQ(backend_kind::multicore, d.group(1).backend);
#ifdef ARB_GPU_ENABLED
        EXPECT_EQ(backend_kind::gpu, d.group(2).backend);
#else
        EXPECT_EQ(backend_kind::multicore, d.group(2).backend);
#endif
        for (unsigned i = 0; i < cells_per_rank*nranks; ++i) {
            EXPECT_EQ((int)(i/cells_per_rank), d.gid_domain(i));
        }
    }
}

TEST(domain_decomposition, invalid)
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

    {
        auto rec = homo_recipe(10*nranks, dummy_cell{});
        std::vector<group_description> groups =
                {{cell_kind::cable, {0, 1, 2, 3, 4, 5, 6, 7, 8, (unsigned)nranks*10}, backend_kind::multicore}};
        EXPECT_THROW(domain_decomposition(rec, ctx, groups), out_of_bounds);

        std::vector<cell_gid_type> gids;
        for (unsigned i = 0; i < 10; ++i) {
            gids.push_back(rank*10 + i);
        }
        if (rank == 0) gids.back() = 0;
        groups = {{cell_kind::cable, gids, backend_kind::multicore}};
        EXPECT_THROW(domain_decomposition(rec, ctx, groups), duplicate_gid);
    }
}
