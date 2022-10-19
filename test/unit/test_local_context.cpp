#include <vector>

#include <gtest/gtest.h>
#include "distributed_context.hpp"

#include <arbor/spike.hpp>

// Test that there are no errors constructing a distributed_context from a local_context
TEST(local_context, construct_distributed_context)
{
    arb::distributed_context ctx = arb::local_context();
}

TEST(local_context, size_rank)
{
    arb::local_context ctx;

    EXPECT_EQ(ctx.size(), 1);
    EXPECT_EQ(ctx.id(), 0);
}

TEST(local_context, minmax)
{
    arb::local_context ctx;

    EXPECT_EQ(1., ctx.min(1.));
    EXPECT_EQ(1., ctx.max(1.));

    EXPECT_EQ(1.f, ctx.min(1.f));
    EXPECT_EQ(1.f, ctx.max(1.f));

    int32_t one32 = 1;
    EXPECT_EQ(one32, ctx.min(one32));
    EXPECT_EQ(one32, ctx.max(one32));

    int64_t one64 = 1;
    EXPECT_EQ(one64, ctx.min(one64));
    EXPECT_EQ(one64, ctx.max(one64));
}

TEST(local_context, sum)
{
    arb::local_context ctx;

    EXPECT_EQ(42.,  ctx.min(42.));
    EXPECT_EQ(42.f, ctx.min(42.));
    EXPECT_EQ(42,   ctx.sum(42));
    EXPECT_EQ(42u,  ctx.min(42u));
}

TEST(local_context, gather)
{
    arb::local_context ctx;

    EXPECT_EQ(std::vector<int>{42}, ctx.gather(42, 0));
    EXPECT_EQ(std::vector<double>{42}, ctx.gather(42., 0));
    EXPECT_EQ(std::vector<std::string>{"42"}, ctx.gather(std::string("42"), 0));
}

TEST(local_context, gather_spikes)
{
    arb::local_context ctx;
    using svec = std::vector<arb::spike>;

    svec spikes = {
        {{0u,3u}, 42.f},
        {{1u,2u}, 42.f},
        {{2u,1u}, 42.f},
        {{3u,0u}, 42.f},
    };

    auto s = ctx.gather_spikes(spikes);

    auto& part = s.partition();
    EXPECT_EQ(s.values(), spikes);
    EXPECT_EQ(part.size(), 2u);
    EXPECT_EQ(part[0], 0u);
    EXPECT_EQ(part[1], spikes.size());
}

TEST(local_context, gather_gids)
{
    arb::local_context ctx;
    using gvec = std::vector<arb::cell_gid_type>;

    gvec gids = {0, 1, 2, 3, 4};

    auto s = ctx.gather_gids(gids);

    auto& part = s.partition();
    EXPECT_EQ(s.values(), gids);
    EXPECT_EQ(part.size(), 2u);
    EXPECT_EQ(part[0], 0u);
    EXPECT_EQ(part[1], gids.size());
}
