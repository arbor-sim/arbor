#include <vector>
#include <cstring>

#include "../gtest.h"

#include <distributed_context.hpp>
#include <arbor/spike.hpp>

// Test that there are no errors constructing a distributed_context from a dry_run_context
using distributed_context_handle = std::shared_ptr<arb::distributed_context>;
int num_ranks = 100;
unsigned num_cells_per_rank = 1000;

TEST(dry_run_context, construct_distributed_context)
{
    distributed_context_handle ctx = arb::make_dryrun_context(num_ranks, num_cells_per_rank);
}

TEST(dry_run_context, size_rank)
{
    distributed_context_handle ctx = arb::make_dryrun_context(num_ranks, num_cells_per_rank);

    EXPECT_EQ(ctx->size(), num_ranks);
    EXPECT_EQ(ctx->id(), 0);
}

TEST(dry_run_context, minmax)
{
    distributed_context_handle ctx = arb::make_dryrun_context(num_ranks, num_cells_per_rank);

    EXPECT_EQ(1., ctx->min(1.));
    EXPECT_EQ(1., ctx->max(1.));

    EXPECT_EQ(1.f, ctx->min(1.f));
    EXPECT_EQ(1.f, ctx->max(1.f));

    int32_t one32 = 1;
    EXPECT_EQ(one32, ctx->min(one32));
    EXPECT_EQ(one32, ctx->max(one32));

    int64_t one64 = 1;
    EXPECT_EQ(one64, ctx->min(one64));
    EXPECT_EQ(one64, ctx->max(one64));
}

TEST(dry_run_context, sum)
{
    distributed_context_handle ctx = arb::make_dryrun_context(num_ranks, num_cells_per_rank);

    EXPECT_EQ(42.,  ctx->min(42.));
    EXPECT_EQ(42.f, ctx->min(42.));
    EXPECT_EQ(42 * num_ranks,   ctx->sum(42));
    EXPECT_EQ(42u,  ctx->min(42u));
}

TEST(dry_run_context, gather_spikes)
{
    distributed_context_handle ctx = arb::make_dryrun_context(4, 4);
    using svec = std::vector<arb::spike>;

    svec spikes = {
        {{0u,3u}, 42.f},
        {{1u,2u}, 42.f},
        {{2u,1u}, 42.f},
        {{3u,0u}, 42.f},
    };
    svec gathered_spikes = {
        {{0u,3u}, 42.f},
        {{1u,2u}, 42.f},
        {{2u,1u}, 42.f},
        {{3u,0u}, 42.f},
        {{4u,3u}, 42.f},
        {{5u,2u}, 42.f},
        {{6u,1u}, 42.f},
        {{7u,0u}, 42.f},
        {{8u,3u}, 42.f},
        {{9u,2u}, 42.f},
        {{10u,1u}, 42.f},
        {{11u,0u}, 42.f},
        {{12u,3u}, 42.f},
        {{13u,2u}, 42.f},
        {{14u,1u}, 42.f},
        {{15u,0u}, 42.f},
    };

    auto s = ctx->gather_spikes(spikes);
    auto& part = s.partition();

    EXPECT_EQ(s.values(), gathered_spikes);
    EXPECT_EQ(part.size(), 5u);
    EXPECT_EQ(part[0], 0u);
    EXPECT_EQ(part[1], spikes.size());
    EXPECT_EQ(part[2], spikes.size()*2);
    EXPECT_EQ(part[3], spikes.size()*3);
    EXPECT_EQ(part[4], spikes.size()*4);
}
