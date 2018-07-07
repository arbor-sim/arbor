#include "../gtest.h"

#include <arbor/util/lexcmp_def.hpp>

struct lexcmp_test_one {
    int foo;
};

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(lexcmp_test_one, (a.foo), (b.foo))

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
ARB_DEFINE_LEXICOGRAPHIC_ORDERING(lexcmp_test_three, (a.z,a.y,a.x), (b.z,b.y,b.x))

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

// test fields accessed by reference-returning member function

class lexcmp_test_refmemfn {
public:
    explicit lexcmp_test_refmemfn(int foo): foo_(foo) {}

    const int& foo() const { return foo_; }
    int& foo() { return foo_; }

private:
    int foo_;
};

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(lexcmp_test_refmemfn, (a.foo()), (b.foo()))

TEST(lexcmp_def,refmemfn) {
    lexcmp_test_refmemfn p{3};
    const lexcmp_test_refmemfn q{4};

    EXPECT_LE(p,q);
    EXPECT_LT(p,q);
    EXPECT_NE(p,q);
    EXPECT_GE(q,p);
    EXPECT_GT(q,p);
}

// test comparison via proxy tuple object

class lexcmp_test_valmemfn {
public:
    explicit lexcmp_test_valmemfn(int foo, int bar): foo_(foo), bar_(bar) {}
    int foo() const { return foo_; }
    int bar() const { return bar_; }

private:
    int foo_;
    int bar_;
};

ARB_DEFINE_LEXICOGRAPHIC_ORDERING_BY_VALUE(lexcmp_test_valmemfn, (a.foo(),a.bar()), (b.foo(),b.bar()))

TEST(lexcmp_def,proxy) {
    lexcmp_test_valmemfn p{3,2}, q{3,4};

    EXPECT_LE(p,q);
    EXPECT_LT(p,q);
    EXPECT_NE(p,q);
    EXPECT_GE(q,p);
    EXPECT_GT(q,p);
}


