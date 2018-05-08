#include <vector>

#include "../gtest.h"
#include <communication/distributed_context.hpp>
#include <spike.hpp>

// Test that there are no errors constructing a distributed_context from a serial_context
TEST(serial_context, construct_distributed_context)
{
    arb::distributed_context ctx = arb::serial_context();
}

TEST(serial_context, size_rank)
{
    arb::serial_context ctx;

    EXPECT_EQ(ctx.size(), 1);
    EXPECT_EQ(ctx.id(), 0);
}

TEST(serial_context, minmax)
{
    arb::serial_context ctx;

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

TEST(serial_context, sum)
{
    arb::serial_context ctx;

    EXPECT_EQ(42.,  ctx.min(42.));
    EXPECT_EQ(42.f, ctx.min(42.));
    EXPECT_EQ(42,   ctx.sum(42));
    EXPECT_EQ(42u,  ctx.min(42u));
}

TEST(serial_context, gather)
{
    arb::serial_context ctx;

    EXPECT_EQ(std::vector<int>{42}, ctx.gather(42, 0));
    EXPECT_EQ(std::vector<double>{42}, ctx.gather(42., 0));
}

TEST(serial_context, gather_spikes)
{
    arb::serial_context ctx;
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
