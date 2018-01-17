#include "../gtest.h"

#include <stdexcept>

#include <backends.hpp>
#include <domain_decomposition.hpp>
#include <hardware/node_info.hpp>
#include <load_balance.hpp>

#include "../simple_recipes.hpp"

using namespace arb;

namespace {
    // Dummy recipes types for testing.

    struct dummy_cell {};
    using homo_recipe = homogeneous_recipe<cell_kind::cable1d_neuron, dummy_cell>;

    // Heterogenous cell population of cable and rss cells.
    // Interleaved so that cells with even gid are cable cells, and odd gid are
    // rss cells.
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
                cell_kind::regular_spike_source:
                cell_kind::cable1d_neuron;
        }

    private:
        cell_size_type size_;
    };
}

TEST(domain_decomposition, homogenous_population)
{
    {   // Test on a node with 1 cpu core and no gpus.
        // We assume that all cells will be put into cell groups of size 1.
        // This assumption will not hold in the future, requiring and update to
        // the test.
        hw::node_info nd(1, 0);

        unsigned num_cells = 10;
        const auto D = partition_load_balance(homo_recipe(num_cells, dummy_cell{}), nd);

        EXPECT_EQ(D.num_global_cells, num_cells);
        EXPECT_EQ(D.num_local_cells, num_cells);
        EXPECT_EQ(D.groups.size(), num_cells);

        auto gids = util::make_span(0, num_cells);
        for (auto gid: gids) {
            EXPECT_EQ(0, D.gid_domain(gid));
            EXPECT_TRUE(D.is_local_gid(gid));
        }
        EXPECT_FALSE(D.is_local_gid(num_cells));

        // Each cell group contains 1 cell of kind cable1d_neuron
        // Each group should also be tagged for cpu execution
        for (auto i: gids) {
            auto& grp = D.groups[i];
            EXPECT_EQ(grp.gids.size(), 1u);
            EXPECT_EQ(grp.gids.front(), unsigned(i));
            EXPECT_EQ(grp.backend, backend_kind::multicore);
            EXPECT_EQ(grp.kind, cell_kind::cable1d_neuron);
        }
    }
    {   // Test on a node with 1 gpu and 1 cpu core.
        // Assumes that all cells will be placed on gpu in a single group.
        hw::node_info nd(1, 1);

        unsigned num_cells = 10;
        const auto D = partition_load_balance(homo_recipe(num_cells, dummy_cell{}), nd);

        EXPECT_EQ(D.num_global_cells, num_cells);
        EXPECT_EQ(D.num_local_cells, num_cells);
        EXPECT_EQ(D.groups.size(), 1u);

        auto gids = util::make_span(0, num_cells);
        for (auto gid: gids) {
            EXPECT_EQ(0, D.gid_domain(gid));
            EXPECT_TRUE(D.is_local_gid(gid));
        }
        EXPECT_FALSE(D.is_local_gid(num_cells));

        // Each cell group contains 1 cell of kind cable1d_neuron
        // Each group should also be tagged for cpu execution
        auto grp = D.groups[0u];

        EXPECT_EQ(grp.gids.size(), num_cells);
        EXPECT_EQ(grp.gids.front(), 0u);
        EXPECT_EQ(grp.gids.back(), num_cells-1);
        EXPECT_EQ(grp.backend, backend_kind::gpu);
        EXPECT_EQ(grp.kind, cell_kind::cable1d_neuron);
    }
}

TEST(domain_decomposition, heterogenous_population)
{
    {   // Test on a node with 1 cpu core and no gpus.
        // We assume that all cells will be put into cell groups of size 1.
        // This assumption will not hold in the future, requiring and update to
        // the test.
        hw::node_info nd(1, 0);

        unsigned num_cells = 10;
        auto R = hetero_recipe(num_cells);
        const auto D = partition_load_balance(R, nd);

        EXPECT_EQ(D.num_global_cells, num_cells);
        EXPECT_EQ(D.num_local_cells, num_cells);
        EXPECT_EQ(D.groups.size(), num_cells);

        auto gids = util::make_span(0, num_cells);
        for (auto gid: gids) {
            EXPECT_EQ(0, D.gid_domain(gid));
            EXPECT_TRUE(D.is_local_gid(gid));
        }
        EXPECT_FALSE(D.is_local_gid(num_cells));

        // Each cell group contains 1 cell of kind cable1d_neuron
        // Each group should also be tagged for cpu execution
        auto grps = util::make_span(0, num_cells);
        std::map<cell_kind, std::set<cell_gid_type>> kind_lists;
        for (auto i: grps) {
            auto& grp = D.groups[i];
            EXPECT_EQ(grp.gids.size(), 1u);
            auto k = grp.kind;
            kind_lists[k].insert(grp.gids.front());
            EXPECT_EQ(grp.backend, backend_kind::multicore);
        }

        for (auto k: {cell_kind::cable1d_neuron, cell_kind::regular_spike_source}) {
            const auto& gids = kind_lists[k];
            EXPECT_EQ(gids.size(), num_cells/2);
            for (auto gid: gids) {
                EXPECT_EQ(k, R.get_cell_kind(gid));
            }
        }
    }
    {   // Test on a node with 1 gpu and 1 cpu core.
        // Assumes that calble cells are on gpu in a single group, and
        // rff cells are on cpu in cell groups of size 1
        hw::node_info nd(1, 1);

        unsigned num_cells = 10;
        auto R = hetero_recipe(num_cells);
        const auto D = partition_load_balance(R, nd);

        EXPECT_EQ(D.num_global_cells, num_cells);
        EXPECT_EQ(D.num_local_cells, num_cells);
        // one cell group with num_cells/2 on gpu, and num_cells/2 groups on cpu
        auto expected_groups = num_cells/2+1;
        EXPECT_EQ(D.groups.size(), expected_groups);

        auto grps = util::make_span(0, expected_groups);
        unsigned ncells = 0;
        // iterate over each group and test its properties
        for (auto i: grps) {
            auto& grp = D.groups[i];
            auto k = grp.kind;
            if (k==cell_kind::cable1d_neuron) {
                EXPECT_EQ(grp.backend, backend_kind::gpu);
                EXPECT_EQ(grp.gids.size(), num_cells/2);
                for (auto gid: grp.gids) {
                    EXPECT_TRUE(gid%2==0);
                    ++ncells;
                }
            }
            else if (k==cell_kind::regular_spike_source){
                EXPECT_EQ(grp.backend, backend_kind::multicore);
                EXPECT_EQ(grp.gids.size(), 1u);
                EXPECT_TRUE(grp.gids.front()%2);
                ++ncells;
            }
        }
        EXPECT_EQ(num_cells, ncells);
    }
}
