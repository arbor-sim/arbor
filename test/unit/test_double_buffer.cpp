#include "../gtest.h"

#include <util/double_buffer.hpp>

// not much to test here: just test that values passed into the constructor
// are correctly stored in members
TEST(double_buffer, exchange_and_get)
{
    using namespace arb::util;

    double_buffer<int> buf;

    buf.get() = 2134;
    buf.exchange();
    buf.get() = 8990;
    buf.exchange();

    EXPECT_EQ(buf.get(), 2134);
    EXPECT_EQ(buf.other(), 8990);
    buf.exchange();
    EXPECT_EQ(buf.get(), 8990);
    EXPECT_EQ(buf.other(), 2134);
    buf.exchange();
    EXPECT_EQ(buf.get(), 2134);
    EXPECT_EQ(buf.other(), 8990);
}

TEST(double_buffer, assign_get_other)
{
    using namespace arb::util;

    double_buffer<std::string> buf;

    buf.get()   = "1";
    buf.other() = "2";

    EXPECT_EQ(buf.get(), "1");
    EXPECT_EQ(buf.other(), "2");
}

TEST(double_buffer, non_pod)
{
    using namespace arb::util;

    double_buffer<std::string> buf;

    buf.get()   = "1";
    buf.other() = "2";

    EXPECT_EQ(buf.get(), "1");
    EXPECT_EQ(buf.other(), "2");
    buf.exchange();
    EXPECT_EQ(buf.get(), "2");
    EXPECT_EQ(buf.other(), "1");
    buf.exchange();
    EXPECT_EQ(buf.get(), "1");
    EXPECT_EQ(buf.other(), "2");
}
