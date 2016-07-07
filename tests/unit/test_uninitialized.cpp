#include "gtest.h"

#include "util/uninitialized.hpp"

using namespace nest::mc::util;

namespace {
    struct count_ops {
        count_ops() {}
        count_ops(const count_ops &n) { ++copy_ctor_count; }
        count_ops(count_ops &&n) { ++move_ctor_count; }

        count_ops &operator=(const count_ops &n) { ++copy_assign_count; return *this; }
        count_ops &operator=(count_ops &&n) { ++move_assign_count; return *this; }

        static int copy_ctor_count,copy_assign_count;
        static int move_ctor_count,move_assign_count;
        static void reset_counts() {
            copy_ctor_count=copy_assign_count=0; 
            move_ctor_count=move_assign_count=0;
        }
    };

    int count_ops::copy_ctor_count=0;
    int count_ops::copy_assign_count=0;
    int count_ops::move_ctor_count=0;
    int count_ops::move_assign_count=0;
}

TEST(uninitialized,ctor) {
    count_ops::reset_counts();

    uninitialized<count_ops> ua;
    ua.construct(count_ops{});

    count_ops b;
    ua.construct(b);

    EXPECT_EQ(1,count_ops::copy_ctor_count);
    EXPECT_EQ(0,count_ops::copy_assign_count);
    EXPECT_EQ(1,count_ops::move_ctor_count);
    EXPECT_EQ(0,count_ops::move_assign_count);

    ua.ref()=count_ops{};
    ua.ref()=b;

    EXPECT_EQ(1,count_ops::copy_ctor_count);
    EXPECT_EQ(1,count_ops::copy_assign_count);
    EXPECT_EQ(1,count_ops::move_ctor_count);
    EXPECT_EQ(1,count_ops::move_assign_count);
}

namespace {
    struct nocopy {
        nocopy() {}
        nocopy(const nocopy &n) = delete;
        nocopy(nocopy &&n) { ++move_ctor_count; }

        nocopy &operator=(const nocopy &n) = delete;
        nocopy &operator=(nocopy &&n) { ++move_assign_count; return *this; }

        static int move_ctor_count,move_assign_count;
        static void reset_counts() { move_ctor_count=move_assign_count=0; }
    };

    int nocopy::move_ctor_count=0;
    int nocopy::move_assign_count=0;
}

TEST(uninitialized,ctor_nocopy) {
    nocopy::reset_counts();

    uninitialized<nocopy> ua;
    ua.construct(nocopy{});

    EXPECT_EQ(1,nocopy::move_ctor_count);
    EXPECT_EQ(0,nocopy::move_assign_count);

    ua.ref()=nocopy{};

    EXPECT_EQ(1,nocopy::move_ctor_count);
    EXPECT_EQ(1,nocopy::move_assign_count);
}

namespace {
    struct nomove {
        nomove() {}
        nomove(const nomove &n) { ++copy_ctor_count; }
        nomove(nomove &&n) = delete;

        nomove &operator=(const nomove &n) { ++copy_assign_count; return *this; }
        nomove &operator=(nomove &&n) = delete;

        static int copy_ctor_count,copy_assign_count;
        static void reset_counts() { copy_ctor_count=copy_assign_count=0; }
    };

    int nomove::copy_ctor_count=0;
    int nomove::copy_assign_count=0;
}

TEST(uninitialized,ctor_nomove) {
    nomove::reset_counts();

    uninitialized<nomove> ua;
    ua.construct(nomove{}); // check against rvalue

    nomove b;
    ua.construct(b); // check against non-const lvalue

    const nomove c;
    ua.construct(c); // check against const lvalue

    EXPECT_EQ(3,nomove::copy_ctor_count);
    EXPECT_EQ(0,nomove::copy_assign_count);

    nomove a;
    ua.ref()=a;

    EXPECT_EQ(3,nomove::copy_ctor_count);
    EXPECT_EQ(1,nomove::copy_assign_count);
}

TEST(uninitialized,void) {
    uninitialized<void> a,b;
    a=b;

    EXPECT_EQ(typeid(a.ref()),typeid(void));
}

TEST(uninitialized,ref) {
    uninitialized<int &> x,y;
    int a;

    x.construct(a);
    y=x;

    x.ref()=2;
    EXPECT_EQ(2,a);

    y.ref()=3;
    EXPECT_EQ(3,a);
    EXPECT_EQ(3,x.cref());

    EXPECT_EQ(&a,x.ptr());
    EXPECT_EQ((const int *)&a,x.cptr());
}

namespace {
    struct apply_tester {
        mutable int op_count=0;
        mutable int const_op_count=0;

        int operator()(const int &a) const { ++const_op_count; return a+1; }
        int operator()(int &a) const { ++op_count; return ++a; }
    };
}

TEST(uninitialized,apply) {
    uninitialized<int> ua;
    ua.construct(10);

    apply_tester A;
    int r=ua.apply(A);
    EXPECT_EQ(11,ua.cref());
    EXPECT_EQ(11,r);

    uninitialized<int &> ub;
    ub.construct(ua.ref());

    r=ub.apply(A);
    EXPECT_EQ(12,ua.cref());
    EXPECT_EQ(12,r);

    uninitialized<const int &> uc;
    uc.construct(ua.ref());

    r=uc.apply(A);
    EXPECT_EQ(12,ua.cref());
    EXPECT_EQ(13,r);

    const uninitialized<int> ud(ua);

    r=ud.apply(A);
    EXPECT_EQ(12,ua.cref());
    EXPECT_EQ(12,ud.cref());
    EXPECT_EQ(13,r);

    EXPECT_EQ(2,A.op_count);
    EXPECT_EQ(2,A.const_op_count);
}

TEST(uninitialized,void_apply) {
    uninitialized<void> uv;

    auto f=[]() { return 11; };
    EXPECT_EQ(11,uv.apply(f));

    EXPECT_EQ(12.5,uv.apply([]() { return 12.5; }));
}
