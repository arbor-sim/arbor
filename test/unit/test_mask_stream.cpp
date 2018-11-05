#include <sstream>

#include "../gtest.h"

#include <sup/ioutil.hpp>

using sup::mask_stream;

TEST(mask_stream,nomask) {
    // expect mask_stream(true) on a new stream not to change rdbuf.
    std::ostringstream s;
    auto sbuf = s.rdbuf();
    s << mask_stream(true);
    EXPECT_EQ(sbuf, s.rdbuf());
}

TEST(mask_stream,mask) {
    // masked stream should produce no ouptut
    std::ostringstream s;
    s << "one";
    s << mask_stream(false);

    s << "two";
    EXPECT_EQ(s.str(), "one");

    s << mask_stream(true);
    s << "three";
    EXPECT_EQ(s.str(), "onethree");
}

TEST(mask_stream,mask_multi) {
    // mark_stream(false) should be idempotent

    std::ostringstream s;
    auto sbuf1 = s.rdbuf();

    s << "foo";
    s << mask_stream(false);
    auto sbuf2 = s.rdbuf();

    s << "bar";
    s << mask_stream(false);
    auto sbuf3 = s.rdbuf();
    EXPECT_EQ(sbuf2, sbuf3);

    s << "baz";
    s << mask_stream(true);
    auto sbuf4 = s.rdbuf();
    EXPECT_EQ(sbuf1, sbuf4);

    s << "xyzzy";
    EXPECT_EQ(s.str(), "fooxyzzy");
}

TEST(mask_stream,fmt) {
    // expect formatting to be preserved across masks.

    std::ostringstream s;
    s.precision(1);

    s << mask_stream(false);
    EXPECT_EQ(s.precision(), 1);
    s.precision(2);

    s << mask_stream(true);
    EXPECT_EQ(s.precision(), 2);
}

