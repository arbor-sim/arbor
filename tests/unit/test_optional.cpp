#include <typeinfo>
#include <array>
#include <algorithm>

#include "gtest.h"
#include "util/optional.hpp"
#include "common.hpp"

using namespace nest::mc::util;

TEST(optionalm,ctors) {
    optional<int> a,b(3),c=b,d=4;

    ASSERT_FALSE((bool)a);
    ASSERT_TRUE((bool)b);
    ASSERT_TRUE((bool)c);
    ASSERT_TRUE((bool)d);

    EXPECT_EQ(3,b.get());
    EXPECT_EQ(3,c.get());
    EXPECT_EQ(4,d.get());
}

TEST(optionalm,unset_throw) {
    optional<int> a;
    int check=10;

    try {
        a.get();
    }
    catch (optional_unset_error& e) {
        ++check;
    }
    EXPECT_EQ(11,check);

    check=20;
    a=2;
    try {
        a.get();
    }
    catch (optional_unset_error& e) {
        ++check;
    }
    EXPECT_EQ(20,check);

    check=30;
    a.reset();
    try {
        a.get();
    }
    catch (optional_unset_error& e) {
        ++check;
    }
    EXPECT_EQ(31,check);
}

TEST(optionalm,deref) {
    struct foo {
        int a;
        explicit foo(int a_): a(a_) {}
        double value() { return 3.0*a; }
    };

    optional<foo> f=foo(2);
    EXPECT_EQ(6.0,f->value());
    EXPECT_EQ(2,(*f).a);
}

TEST(optionalm,ctor_conv) {
    optional<std::array<int,3>> x{{1,2,3}};
    EXPECT_EQ(3u,x->size());
}

TEST(optionalm,ctor_ref) {
    int v=10;
    optional<int&> a(v);

    EXPECT_EQ(10,a.get());
    v=20;
    EXPECT_EQ(20,a.get());

    optional<int&> b(a),c=b,d=v;
    EXPECT_EQ(&(a.get()),&(b.get()));
    EXPECT_EQ(&(a.get()),&(c.get()));
    EXPECT_EQ(&(a.get()),&(d.get()));
}

TEST(optionalm,assign_returns) {
    optional<int> a=3;

    auto b=(a=4);
    EXPECT_EQ(typeid(optional<int>),typeid(b));

    auto bp=&(a=4);
    EXPECT_EQ(&a,bp);

    auto b2=(a=optional<int>(10));
    EXPECT_EQ(typeid(optional<int>),typeid(b2));

    auto bp2=&(a=4);
    EXPECT_EQ(&a,bp2);

    auto b3=(a=nothing);
    EXPECT_EQ(typeid(optional<int>),typeid(b3));

    auto bp3=&(a=4);
    EXPECT_EQ(&a,bp3);
}

TEST(optionalm,assign_reference) {
    double a=3.0;
    optional<double&> ar;
    optional<double&> br;

    ar = a;
    EXPECT_TRUE(ar);
    *ar = 5.0;
    EXPECT_EQ(5.0, a);

    auto& check_rval=(br=ar);
    EXPECT_TRUE(br);
    EXPECT_EQ(&br, &check_rval);

    *br = 7.0;
    EXPECT_EQ(7.0, a);

    auto& check_rval2=(br=nothing);
    EXPECT_FALSE(br);
    EXPECT_EQ(&br, &check_rval2);
}

TEST(optionalm,ctor_nomove) {
    using nomove = testing::nomove<int>;

    optional<nomove> a(nomove(3));
    EXPECT_EQ(nomove(3),a.get());

    optional<nomove> b;
    b=a;
    EXPECT_EQ(nomove(3),b.get());

    b=optional<nomove>(nomove(4));
    EXPECT_EQ(nomove(4),b.get());
}

TEST(optionalm,ctor_nocopy) {
    using nocopy = testing::nocopy<int>;

    optional<nocopy> a(nocopy(5));
    EXPECT_EQ(nocopy(5),a.get());

    nocopy::reset_counts();
    optional<nocopy> b(std::move(a));
    EXPECT_EQ(nocopy(5),b.get());
    EXPECT_EQ(0,a.get().value);
    EXPECT_EQ(1, nocopy::move_ctor_count);
    EXPECT_EQ(0, nocopy::move_assign_count);

    nocopy::reset_counts();
    b=optional<nocopy>(nocopy(6));
    EXPECT_EQ(nocopy(6),b.get());
    EXPECT_EQ(1, nocopy::move_ctor_count);
    EXPECT_EQ(1, nocopy::move_assign_count);

}

static optional<double> odd_half(int n) {
    optional<double> h;
    if (n%2==1) h=n/2.0;
    return h;
}

TEST(optionalm,bind) {
    optional<int> a;
    auto b=a.bind(odd_half);

    EXPECT_EQ(typeid(optional<double>),typeid(b));

    a=10;
    b=a.bind(odd_half);
    EXPECT_FALSE((bool)b);

    a=11;
    b=a.bind(odd_half);
    EXPECT_TRUE((bool)b);
    EXPECT_EQ(5.5,b.get());

    b=a >> odd_half >> [](double x) { return (int)x; } >> odd_half;
    EXPECT_TRUE((bool)b);
    EXPECT_EQ(2.5,b.get());
}

TEST(optionalm,void) {
    optional<void> a,b(true),c(a),d=b,e(false),f(nothing);

    EXPECT_FALSE((bool)a);
    EXPECT_TRUE((bool)b);
    EXPECT_FALSE((bool)c);
    EXPECT_TRUE((bool)d);
    EXPECT_TRUE((bool)e);
    EXPECT_FALSE((bool)f);

    auto x=a >> []() { return 1; };
    EXPECT_FALSE((bool)x);

    x=b >> []() { return 1; };
    EXPECT_TRUE((bool)x);
    EXPECT_EQ(1,x.get());

    auto& check_rval=(b=nothing);
    EXPECT_FALSE((bool)b);
    EXPECT_EQ(&b,&check_rval);
}

TEST(optionalm,bind_to_void) {
    optional<int> a,b(3);

    int call_count=0;
    auto vf=[&call_count](int i) -> void { ++call_count; };

    auto x=a >> vf;
    EXPECT_EQ(typeid(optional<void>),typeid(x));
    EXPECT_FALSE((bool)x);
    EXPECT_EQ(0,call_count);

    call_count=0;
    x=b >> vf;
    EXPECT_TRUE((bool)x);
    EXPECT_EQ(1,call_count);
}

TEST(optionalm,bind_to_optional_void) {
    optional<int> a,b(3),c(4);

    int count=0;
    auto count_if_odd=[&count](int i) {
        return i%2?(++count,optional<void>(true)):optional<void>();
    };

    auto x=a >> count_if_odd;
    EXPECT_EQ(typeid(optional<void>),typeid(x));
    EXPECT_FALSE((bool)x);
    EXPECT_EQ(0,count);

    count=0;
    x=b >> count_if_odd;
    EXPECT_TRUE((bool)x);
    EXPECT_EQ(1,count);

    count=0;
    x=c >> count_if_odd;
    EXPECT_FALSE((bool)x);
    EXPECT_EQ(0,count);
}

TEST(optionalm,bind_with_ref) {
    optional<int> a=10;
    a >> [](int& v) { ++v; };
    EXPECT_EQ(11,*a);
}

struct check_cref {
    int operator()(const int&) { return 10; }
    int operator()(int&) { return 11; }
};

TEST(optionalm,bind_constness) {
    check_cref checker;
    optional<int> a=1;
    int v=*(a >> checker);
    EXPECT_EQ(11,v);

    const optional<int> b=1;
    v=*(b >> checker);
    EXPECT_EQ(10,v);
}


TEST(optionalm,conversion) {
    optional<double> a(3),b=5;
    EXPECT_TRUE((bool)a);
    EXPECT_TRUE((bool)b);
    EXPECT_EQ(3.0,a.get());
    EXPECT_EQ(5.0,b.get());

    optional<int> x;
    optional<double> c(x);
    optional<double> d=optional<int>();
    EXPECT_FALSE((bool)c);
    EXPECT_FALSE((bool)d);

    auto doubler=[](double x) { return x*2; };
    auto y=optional<int>(3) >> doubler;
    EXPECT_TRUE((bool)y);
    EXPECT_EQ(6.0,y.get());
}

TEST(optionalm,or_operator) {
    optional<const char *> default_msg="default";
    auto x=(char *)0 | default_msg;
    EXPECT_TRUE((bool)x);
    EXPECT_STREQ("default",x.get());

    auto y="something" | default_msg;
    EXPECT_TRUE((bool)y);
    EXPECT_STREQ("something",y.get());

    optional<int> a(1),b,c(3);
    EXPECT_EQ(1,*(a|b|c));
    EXPECT_EQ(1,*(a|c|b));
    EXPECT_EQ(1,*(b|a|c));
    EXPECT_EQ(3,*(b|c|a));
    EXPECT_EQ(3,*(c|a|b));
    EXPECT_EQ(3,*(c|b|a));
}

TEST(optionalm,and_operator) {
    optional<int> a(1);
    optional<double> b(2.0);

    auto ab=a&b;
    auto ba=b&a;

    EXPECT_EQ(typeid(ab),typeid(b));
    EXPECT_EQ(typeid(ba),typeid(a));
    EXPECT_EQ(2.0,*ab);
    EXPECT_EQ(1,*ba);

    auto zb=false & b;
    EXPECT_EQ(typeid(zb),typeid(b));
    EXPECT_FALSE((bool)zb);

    auto b3=b & 3;
    EXPECT_EQ(typeid(b3),typeid(optional<int>));
    EXPECT_TRUE((bool)b3);
    EXPECT_EQ(3,*b3);
}

TEST(optionalm,provided) {
    std::array<int,3> qs={1,0,3};
    std::array<int,3> ps={14,14,14};
    std::array<int,3> rs;

    std::transform(ps.begin(),ps.end(),qs.begin(),rs.begin(),
        [](int p,int q) { return *( provided(q!=0) >> [=]() { return p/q; } | -1 ); });

    EXPECT_EQ(14,rs[0]);
    EXPECT_EQ(-1,rs[1]);
    EXPECT_EQ(4,rs[2]);
}
