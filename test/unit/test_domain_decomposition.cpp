#include "../gtest.h"

#include <stdexcept>

#include <arbor/context.hpp>
#include <arbor/domdecexcept.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>

#include <arborenv/default_env.hpp>

#include "util/span.hpp"

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace arb;
using arb::util::make_span;

// TODO
// The tests here will only test domain decomposition with GPUs when compiled
// with CUDA support and run on a system with a GPU.
// Ideally the tests should test domain decompositions under all conditions, however
// to do that we have to refactor the partition_load_balance algorithm.
// The partition_load_balance performs the decomposition to distribute
// over resources described by the user-supplied arb::context, which is a
// provides an interface to resources available at runtime.
// The best way to test under all conditions, is probably to refactor the
// partition_load_balance into components that can be tested in isolation.

namespace {
    // Dummy recipes types for testing.

    struct dummy_cell {};
    using homo_recipe = homogeneous_recipe<cell_kind::cable, dummy_cell>;

    // Heterogenous cell population of cable and spike source cells.
    // Interleaved so that cells with even gid are cable cells, and odd gid are
    // spike source cells.
    class hetero_recipe: public recipe {
    public:
        hetero_recipe(cell_size_type s): size_(s) {}

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

    private:
        cell_size_type size_;
    };

    class gap_recipe: public recipe {
    public:
        gap_recipe() {}

        cell_size_type num_cells() const override {
            return size_;
        }

        arb::util::unique_any get_cell_description(cell_gid_type) const override {
            auto c = arb::make_cell_soma_only(false);
            c.decorations.place(mlocation{0,1}, junction("gj"), "gj");
            return {arb::cable_cell(c)};
        }

        cell_kind get_cell_kind(cell_gid_type gid) const override {
            return cell_kind::cable;
        }
        std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override {
            switch (gid) {
                case 0 :  return {
                    gap_junction_connection({13, "gj"}, {"gj"}, 0.1)
                };
                case 2 :  return {
                    gap_junction_connection({7,  "gj"}, {"gj"}, 0.1),
                    gap_junction_connection({11, "gj"}, {"gj"}, 0.1)
                };
                case 3 :  return {
                    gap_junction_connection({4, "gj"}, {"gj"}, 0.1),
                    gap_junction_connection({8, "gj"}, {"gj"}, 0.1)
                };
                case 4 :  return {
                    gap_junction_connection({3, "gj"}, {"gj"}, 0.1),
                    gap_junction_connection({8, "gj"}, {"gj"}, 0.1),
                    gap_junction_connection({9, "gj"}, {"gj"}, 0.1)
                };
                case 7 :  return {
                    gap_junction_connection({2,  "gj"}, {"gj"}, 0.1),
                    gap_junction_connection({11, "gj"}, {"gj"}, 0.1)
                };;
                case 8 :  return {
                    gap_junction_connection({3, "gj"}, {"gj"}, 0.1),
                    gap_junction_connection({4, "gj"}, {"gj"}, 0.1)
                };;
                case 9 :  return {
                    gap_junction_connection({4, "gj"}, {"gj"}, 0.1)
                };
                case 11 : return {
                    gap_junction_connection({2, "gj"}, {"gj"}, 0.1),
                    gap_junction_connection({7, "gj"}, {"gj"}, 0.1)
                };
                case 13 : return {
                    gap_junction_connection({0, "gj"}, {"gj"}, 0.1)
                };
                default : return {};
            }
        }

    private:
        cell_size_type size_ = 15;
    };
}

// test assumes one domain
TEST(domain_decomposition, homogenous_population)
{
    proc_allocation resources;
    resources.gpu_id = arbenv::default_gpu();

    if (resources.has_gpu()) {
        // Test on a node with 1 gpu and 1 cpu core.
        // Assumes that all cells will be placed on gpu in a single group.
        auto ctx = make_context(resources);

        unsigned num_cells = 10;
        auto rec = homo_recipe(num_cells, dummy_cell{});
        const auto D = partition_load_balance(rec, ctx);
        EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, D));

        EXPECT_EQ(D.num_global_cells, num_cells);
        EXPECT_EQ(D.num_local_cells, num_cells);
        EXPECT_EQ(D.groups.size(), 1u);

        auto gids = make_span(num_cells);
        for (auto gid: gids) {
            EXPECT_EQ(0, D.gid_domain(gid));
        }

        // Each cell group contains 1 cell of kind cable
        // Each group should also be tagged for cpu execution
        auto grp = D.groups[0u];

        EXPECT_EQ(grp.gids.size(), num_cells);
        EXPECT_EQ(grp.gids.front(), 0u);
        EXPECT_EQ(grp.gids.back(), num_cells-1);
        EXPECT_EQ(grp.backend, backend_kind::gpu);
        EXPECT_EQ(grp.kind, cell_kind::cable);
    }
    {
        resources.gpu_id = -1; // disable GPU if available
        auto ctx = make_context(resources);

        // Test on a node with 1 cpu core and no gpus.
        // We assume that all cells will be put into cell groups of size 1.
        // This assumption will not hold in the future, requiring and update to
        // the test.

        unsigned num_cells = 10;
        auto rec = homo_recipe(num_cells, dummy_cell{});
        const auto D = partition_load_balance(rec, ctx);
        EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, D));

        EXPECT_EQ(D.num_global_cells, num_cells);
        EXPECT_EQ(D.num_local_cells, num_cells);
        EXPECT_EQ(D.groups.size(), num_cells);

        auto gids = make_span(num_cells);
        for (auto gid: gids) {
            EXPECT_EQ(0, D.gid_domain(gid));
        }

        // Each cell group contains 1 cell of kind cable
        // Each group should also be tagged for cpu execution
        for (auto i: gids) {
            auto& grp = D.groups[i];
            EXPECT_EQ(grp.gids.size(), 1u);
            EXPECT_EQ(grp.gids.front(), unsigned(i));
            EXPECT_EQ(grp.backend, backend_kind::multicore);
            EXPECT_EQ(grp.kind, cell_kind::cable);
        }
    }
}

TEST(domain_decomposition, heterogenous_population)
{
    proc_allocation resources;
    resources.gpu_id = arbenv::default_gpu();

    if (resources.has_gpu()) {
        // Test on a node with 1 gpu and 1 cpu core.
        // Assumes that calble cells are on gpu in a single group, and
        // rff cells are on cpu in cell groups of size 1
        auto ctx = make_context(resources);

        unsigned num_cells = 10;
        auto R = hetero_recipe(num_cells);
        const auto D = partition_load_balance(R, ctx);
        EXPECT_NO_THROW(check_domain_decomposition(R, *ctx, D));

        EXPECT_EQ(D.num_global_cells, num_cells);
        EXPECT_EQ(D.num_local_cells, num_cells);
        // one cell group with num_cells/2 on gpu, and num_cells/2 groups on cpu
        auto expected_groups = num_cells/2+1;
        EXPECT_EQ(D.groups.size(), expected_groups);

        auto grps = make_span(expected_groups);
        unsigned ncells = 0;
        // iterate over each group and test its properties
        for (auto i: grps) {
            auto& grp = D.groups[i];
            auto k = grp.kind;
            if (k==cell_kind::cable) {
                EXPECT_EQ(grp.backend, backend_kind::gpu);
                EXPECT_EQ(grp.gids.size(), num_cells/2);
                for (auto gid: grp.gids) {
                    EXPECT_TRUE(gid%2==0);
                    ++ncells;
                }
            }
            else if (k==cell_kind::spike_source){
                EXPECT_EQ(grp.backend, backend_kind::multicore);
                EXPECT_EQ(grp.gids.size(), 1u);
                EXPECT_TRUE(grp.gids.front()%2);
                ++ncells;
            }
        }
        EXPECT_EQ(num_cells, ncells);
    }
    {
        // Test on a node with 1 cpu core and no gpus.
        // We assume that all cells will be put into cell groups of size 1.
        // This assumption will not hold in the future, requiring and update to
        // the test.

        resources.gpu_id = -1; // disable GPU if available
        auto ctx = make_context(resources);

        unsigned num_cells = 10;
        auto R = hetero_recipe(num_cells);
        const auto D = partition_load_balance(R, ctx);

        EXPECT_EQ(D.num_global_cells, num_cells);
        EXPECT_EQ(D.num_local_cells, num_cells);
        EXPECT_EQ(D.groups.size(), num_cells);

        auto gids = make_span(num_cells);
        for (auto gid: gids) {
            EXPECT_EQ(0, D.gid_domain(gid));
        }

        // Each cell group contains 1 cell of kind cable
        // Each group should also be tagged for cpu execution
        auto grps = make_span(num_cells);
        std::map<cell_kind, std::set<cell_gid_type>> kind_lists;
        for (auto i: grps) {
            auto& grp = D.groups[i];
            EXPECT_EQ(grp.gids.size(), 1u);
            auto k = grp.kind;
            kind_lists[k].insert(grp.gids.front());
            EXPECT_EQ(grp.backend, backend_kind::multicore);
        }

        for (auto k: {cell_kind::cable, cell_kind::spike_source}) {
            const auto& gids = kind_lists[k];
            EXPECT_EQ(gids.size(), num_cells/2);
            for (auto gid: gids) {
                EXPECT_EQ(k, R.get_cell_kind(gid));
            }
        }
    }
}

TEST(domain_decomposition, hints) {
    // Check that we can provide group size hint and gpu/cpu preference
    // by cell kind.
    // The hints perfer the multicore backend, so the decomposition is expected
    // to never have cell groups on the GPU, regardless of whether a GPU is
    // available or not.

    auto ctx = make_context();

    partition_hint_map hints;
    hints[cell_kind::cable].cpu_group_size = 3;
    hints[cell_kind::cable].prefer_gpu = false;
    hints[cell_kind::spike_source].cpu_group_size = 4;

    auto rec = hetero_recipe(20);
    domain_decomposition D = partition_load_balance(rec, ctx, hints);
    EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, D));


    std::vector<std::vector<cell_gid_type>> expected_c1d_groups =
        {{0, 2, 4}, {6, 8, 10}, {12, 14, 16}, {18}};

    std::vector<std::vector<cell_gid_type>> expected_ss_groups =
        {{1, 3, 5, 7}, {9, 11, 13, 15}, {17, 19}};

    std::vector<std::vector<cell_gid_type>> c1d_groups, ss_groups;

    for (auto& g: D.groups) {
        EXPECT_TRUE(g.kind==cell_kind::cable || g.kind==cell_kind::spike_source);

        if (g.kind==cell_kind::cable) {
            c1d_groups.push_back(g.gids);
        }
        else if (g.kind==cell_kind::spike_source) {
            ss_groups.push_back(g.gids);
        }
    }

    EXPECT_EQ(expected_c1d_groups, c1d_groups);
    EXPECT_EQ(expected_ss_groups, ss_groups);
}

TEST(domain_decomposition, compulsory_groups) {
    proc_allocation resources;
    resources.num_threads = 1;
    resources.gpu_id = -1;
    auto ctx = make_context(resources);

    auto R = gap_recipe();
    const auto D0 = partition_load_balance(R, ctx);
    EXPECT_NO_THROW(check_domain_decomposition(R, *ctx, D0));

    EXPECT_EQ(9u, D0.groups.size());

    std::vector<std::vector<cell_gid_type>> expected_groups0 =
            { {1}, {5}, {6}, {10}, {12}, {14}, {0, 13}, {2, 7, 11}, {3, 4, 8, 9} };

    for (unsigned i = 0; i < 9u; i++) {
        EXPECT_EQ(expected_groups0[i], D0.groups[i].gids);
    }

    // Test different group_hints
    partition_hint_map hints;
    hints[cell_kind::cable].cpu_group_size = 3;
    hints[cell_kind::cable].prefer_gpu = false;

    const auto D1 = partition_load_balance(R, ctx, hints);
    EXPECT_NO_THROW(check_domain_decomposition(R, *ctx, D1));
    EXPECT_EQ(5u, D1.groups.size());

    std::vector<std::vector<cell_gid_type>> expected_groups1 =
            { {1, 5, 6}, {10, 12, 14}, {0, 13}, {2, 7, 11}, {3, 4, 8, 9} };

    for (unsigned i = 0; i < 5u; i++) {
        EXPECT_EQ(expected_groups1[i], D1.groups[i].gids);
    }

    hints[cell_kind::cable].cpu_group_size = 20;
    hints[cell_kind::cable].prefer_gpu = false;

    const auto D2 = partition_load_balance(R, ctx, hints);
    EXPECT_EQ(1u, D2.groups.size());

    std::vector<cell_gid_type> expected_groups2 =
            { 1, 5, 6, 10, 12, 14, 0, 13, 2, 7, 11, 3, 4, 8, 9 };

    EXPECT_EQ(expected_groups2, D2.groups[0].gids);

}

TEST(domain_decomposition, partition_by_groups) {
    proc_allocation resources;
    resources.num_threads = 1;
    resources.gpu_id = arbenv::default_gpu();
    auto ctx = make_context(resources);

    {
        const unsigned ncells = 10;
        auto rec = homo_recipe(ncells, dummy_cell{});
        std::vector<cell_gid_type> gids(ncells);
        std::iota(gids.begin(), gids.end(), 0);

#ifdef ARB_GPU_ENABLED
        auto d = partition_by_group(rec, ctx, {{cell_kind::cable, gids, backend_kind::gpu}});
#else
        auto d = partition_by_group(rec, ctx, {{cell_kind::cable, gids, backend_kind::multicore}});
#endif
        EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, d));

        EXPECT_EQ(1, d.num_domains);
        EXPECT_EQ(0, d.domain_id);
        EXPECT_EQ(ncells, d.num_local_cells);
        EXPECT_EQ(ncells, d.num_global_cells);
        EXPECT_EQ(1u,     d.groups.size());
        EXPECT_EQ(gids,   d.groups.front().gids);
        EXPECT_EQ(cell_kind::cable, d.groups.front().kind);
#ifdef ARB_GPU_ENABLED
        EXPECT_EQ(backend_kind::gpu, d.groups.front().backend);
#else
        EXPECT_EQ(backend_kind::multicore, d.groups.front().backend);
#endif
        for (unsigned i = 0; i < ncells; ++i) {
            EXPECT_EQ(0, d.gid_domain(i));
        }
    }
    {
        const unsigned ncells = 10;
        auto rec = homo_recipe(ncells, dummy_cell{});
        std::vector<group_description> groups;
        for (unsigned i = 0; i < ncells; ++i) {
            groups.push_back({cell_kind::cable, {i}, backend_kind::multicore});
        }
        auto d = partition_by_group(rec, ctx, groups);

        EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, d));

        EXPECT_EQ(1, d.num_domains);
        EXPECT_EQ(0, d.domain_id);
        EXPECT_EQ(ncells, d.num_local_cells);
        EXPECT_EQ(ncells, d.num_global_cells);
        EXPECT_EQ(ncells, d.groups.size());
        for (unsigned i = 0; i < ncells; ++i) {
            EXPECT_EQ(std::vector<cell_gid_type>{i}, d.groups[i].gids);
            EXPECT_EQ(cell_kind::cable, d.groups[i].kind);
            EXPECT_EQ(backend_kind::multicore, d.groups[i].backend);
        }
        for (unsigned i = 0; i < ncells; ++i) {
            EXPECT_EQ(0, d.gid_domain(i));
        }
    }
    {
        auto rec = gap_recipe();
        std::vector<cell_gid_type> gids(rec.num_cells());
        std::iota(gids.begin(), gids.end(), 0);
        auto d0 = partition_by_group(rec, ctx,  {{cell_kind::cable, gids, backend_kind::multicore}});
        EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, d0));

        auto d1 = partition_by_group(rec, ctx, {{cell_kind::cable, {0, 13}, backend_kind::multicore},
                                                {cell_kind::cable, {2, 7, 11}, backend_kind::multicore},
                                                {cell_kind::cable, {1, 3, 4, 5, 6, 8, 9, 10, 12, 14}, backend_kind::multicore}});
        EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, d1));
    }
}

TEST(domain_decomposition, invalid) {
    proc_allocation resources;
    resources.num_threads = 1;
    resources.gpu_id = -1; // disable GPU if available
    auto ctx = make_context(resources);

    {
        auto rec = homo_recipe(10, dummy_cell{});
        auto d = domain_decomposition();
        d.num_domains = 2;
        EXPECT_THROW(check_domain_decomposition(rec, *ctx, d), invalid_num_domains);

        d.num_domains = 1;
        d.domain_id = 1;
        EXPECT_THROW(check_domain_decomposition(rec, *ctx, d), invalid_domain_id);

        d.domain_id = 0;
        d.num_local_cells = 12;
        d.groups = {{cell_kind::cable, {0, 1, 2, 3, 4, 5, 6, 7, 8}, backend_kind::multicore}};
        EXPECT_THROW(check_domain_decomposition(rec, *ctx, d), invalid_num_local_cells);

        d.num_local_cells = 10;
        d.groups = {{cell_kind::cable, {0, 1, 2, 3, 4, 5, 6, 7, 8, 10}, backend_kind::multicore}};
        EXPECT_THROW(check_domain_decomposition(rec, *ctx, d), out_of_bounds);

        d.groups = {{cell_kind::cable, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, backend_kind::gpu}};
#ifdef ARB_GPU_ENABLED
        EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, d));

        d.groups = {{cell_kind::lif, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, backend_kind::gpu}};
        EXPECT_THROW(check_domain_decomposition(rec, *ctx, d), incompatible_backend);
#else
        EXPECT_THROW(check_domain_decomposition(rec, *ctx, d), invalid_backend);
#endif

        d.groups = {{cell_kind::cable, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, backend_kind::multicore}};
        d.num_global_cells = 12;
        EXPECT_THROW(check_domain_decomposition(rec, *ctx, d), invalid_num_global_cells);

        d.num_global_cells = 10;
        d.groups = {{cell_kind::cable, {0, 1, 2, 3, 4, 5, 6, 7, 8, 8}, backend_kind::multicore}};
        EXPECT_THROW(check_domain_decomposition(rec, *ctx, d), duplicate_gid);

        d.groups = {{cell_kind::cable, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, backend_kind::multicore}};
        d.gid_domain = [](cell_gid_type) { return 1; };
        EXPECT_THROW(check_domain_decomposition(rec, *ctx, d), non_existent_rank);

        d.gid_domain = [](cell_gid_type) { return 0; };
        EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, d));
    }
    {
        auto rec = gap_recipe();
        auto d = domain_decomposition();
        d.num_domains = 1;
        d.domain_id = 0;
        d.num_local_cells = 15;
        d.num_global_cells = 15;
        d.gid_domain = [](cell_gid_type) { return 0; };
        d.groups = {{cell_kind::cable, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, backend_kind::multicore}};
        EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, d));

        d.groups = {{cell_kind::cable, {0, 13}, backend_kind::multicore},
                    {cell_kind::cable, {2, 7, 11}, backend_kind::multicore},
                    {cell_kind::cable, {1, 3, 4, 5, 6, 8, 9, 10, 12, 14}, backend_kind::multicore}};
        EXPECT_NO_THROW(check_domain_decomposition(rec, *ctx, d));

        d.groups = {{cell_kind::cable, {0}, backend_kind::multicore},
                    {cell_kind::cable, {2, 7, 11, 13}, backend_kind::multicore},
                    {cell_kind::cable, {1, 3, 4, 5, 6, 8, 9, 10, 12, 14}, backend_kind::multicore}};
        EXPECT_THROW(check_domain_decomposition(rec, *ctx, d), invalid_gj_cell_group);
    }
}