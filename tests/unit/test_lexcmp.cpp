#include "gtest.h"

#include <util/lexcmp_def.hpp>

struct lexcmp_test_one {
    int foo;
};

DEFINE_LEXICOGRAPHIC_ORDERING(lexcmp_test_one, (a.foo), (b.foo))

TEST(lexcmp_def,one) {
    lexcmp_test_one p{3}, q{4}, r{4};

    EXPECT_LE(p,q);
    EXPECT_LT(p,q);
    EXPECT_NE(p,q);
    EXPECT_GE(q,p);
    EXPECT_GT(q,p);

    EXPECT_LE(q,r);
    EXPECT_GE(q,r);
    EXPECT_EQ(q,r);
}

struct lexcmp_test_three {
    int x;
    std::string y;
    double z;
};

// test fields in reverse order: z, y, x
DEFINE_LEXICOGRAPHIC_ORDERING(lexcmp_test_three, (a.z,a.y,a.x), (b.z,b.y,b.x))

TEST(lexcmp_def,three) {
    lexcmp_test_three p{1,"foo",2};
    lexcmp_test_three q{1,"foo",3};
    lexcmp_test_three r{1,"bar",2};
    lexcmp_test_three s{5,"foo",2};

    EXPECT_LE(p,q);
    EXPECT_LT(p,q);
    EXPECT_NE(p,q);
    EXPECT_GE(q,p);
    EXPECT_GT(q,p);

    EXPECT_LE(r,p);
    EXPECT_LT(r,p);
    EXPECT_NE(p,r);
    EXPECT_GE(p,r);
    EXPECT_GT(p,r);

    EXPECT_LE(p,s);
    EXPECT_LT(p,s);
    EXPECT_NE(p,s);
    EXPECT_GE(s,p);
    EXPECT_GT(s,p);
}

