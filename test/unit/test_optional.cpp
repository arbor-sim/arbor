#include <algorithm>
#include <array>
#include <string>
#include <typeinfo>

#include "../gtest.h"

#include <arbor/util/optional.hpp>

#include "common.hpp"

using namespace std::string_literals;
using namespace arb::util;

TEST(optional, ctors) {
    optional<int> a, b(3), c = b, d = 4;

    ASSERT_FALSE((bool)a);
    ASSERT_TRUE((bool)b);
    ASSERT_TRUE((bool)c);
    ASSERT_TRUE((bool)d);

    EXPECT_EQ(3, b.value());
    EXPECT_EQ(3, c.value());
    EXPECT_EQ(4, d.value());
}

TEST(optional, unset_throw) {
    optional<int> a;
    int check = 10;

    try {
        a.value();
    }
    catch (optional_unset_error& e) {
        ++check;
    }
    EXPECT_EQ(11, check);

    check = 20;
    a = 2;
    try {
        a.value();
    }
    catch (optional_unset_error& e) {
        ++check;
    }
    EXPECT_EQ(20, check);

    check = 30;
    a.reset();
    try {
        a.value();
    }
    catch (optional_unset_error& e) {
        ++check;
    }
    EXPECT_EQ(31, check);
}

TEST(optional, deref) {
    struct foo {
        int a;
        explicit foo(int a_): a(a_) {}
        double value() { return 3.0*a; }
    };

    optional<foo> f = foo(2);
    EXPECT_EQ(6.0, f->value());
    EXPECT_EQ(2, (*f).a);
}

TEST(optional, ctor_conv) {
    optional<std::array<int, 3>> x{{1, 2, 3}};
    EXPECT_EQ(3u, x->size());
}

TEST(optional, ctor_ref) {
    int v = 10;
    optional<int&> a(v);

    EXPECT_EQ(10, a.value());
    v = 20;
    EXPECT_EQ(20, a.value());

    optional<int&> b(a), c = b, d = v;
    EXPECT_EQ(&(a.value()), &(b.value()));
    EXPECT_EQ(&(a.value()), &(c.value()));
    EXPECT_EQ(&(a.value()), &(d.value()));
}

TEST(optional, assign_returns) {
    optional<int> a = 3;

    auto b = (a = 4);
    EXPECT_EQ(typeid(optional<int>), typeid(b));

    auto bp = &(a = 4);
    EXPECT_EQ(&a, bp);

    auto b2 = (a = optional<int>(10));
    EXPECT_EQ(typeid(optional<int>), typeid(b2));

    auto bp2 = &(a = 4);
    EXPECT_EQ(&a, bp2);

    auto b3 = (a = nullopt);
    EXPECT_EQ(typeid(optional<int>), typeid(b3));

    auto bp3 = &(a = 4);
    EXPECT_EQ(&a, bp3);
}

TEST(optional, assign_reference) {
    double a = 3.0;
    optional<double&> ar;
    optional<double&> br;

    ar = a;
    EXPECT_TRUE(ar);
    *ar = 5.0;
    EXPECT_EQ(5.0, a);

    auto& check_rval = (br = ar);
    EXPECT_TRUE(br);
    EXPECT_EQ(&br, &check_rval);

    *br = 7.0;
    EXPECT_EQ(7.0, a);

    auto& check_rval2 = (br = nullopt);
    EXPECT_FALSE(br);
    EXPECT_EQ(&br, &check_rval2);
}

TEST(optional, ctor_nomove) {
    using nomove = testing::nomove<int>;

    optional<nomove> a(nomove(3));
    EXPECT_EQ(nomove(3), a.value());

    optional<nomove> b;
    b = a;
    EXPECT_EQ(nomove(3), b.value());

    b = optional<nomove>(nomove(4));
    EXPECT_EQ(nomove(4), b.value());
}

TEST(optional, ctor_nocopy) {
    using nocopy = testing::nocopy<int>;

    optional<nocopy> a(nocopy(5));
    EXPECT_EQ(nocopy(5), a.value());

    nocopy::reset_counts();
    optional<nocopy> b(std::move(a));
    EXPECT_EQ(nocopy(5), b.value());
    EXPECT_EQ(0, a.value().value);
    EXPECT_EQ(1, nocopy::move_ctor_count);
    EXPECT_EQ(0, nocopy::move_assign_count);

    nocopy::reset_counts();
    b = optional<nocopy>(nocopy(6));
    EXPECT_EQ(nocopy(6), b.value());
    EXPECT_EQ(1, nocopy::move_ctor_count);
    EXPECT_EQ(1, nocopy::move_assign_count);

    nocopy::reset_counts();
    nocopy v = optional<nocopy>(nocopy(9)).value();
    EXPECT_EQ(2, nocopy::move_ctor_count);
    EXPECT_EQ(nocopy(9), v.value);

    const optional<nocopy> ccheck(nocopy(1));
    EXPECT_TRUE(std::is_rvalue_reference<decltype(std::move(ccheck).value())>::value);
    EXPECT_TRUE(std::is_const<std::remove_reference_t<decltype(std::move(ccheck).value())>>::value);
}

TEST(optional, value_or) {
    optional<double> x = 3;
    EXPECT_EQ(3., x.value_or(5));

    x = nullopt;
    EXPECT_EQ(5., x.value_or(5));

    // `value_or` returns T for optional<T>:
    struct check_conv {
        bool value = false;
        explicit check_conv(bool value): value(value) {}

        explicit operator std::string() const {
            return value? "true": "false";
        }
    };
    check_conv cc{true};

    optional<std::string> present = "present"s;
    optional<std::string> absent; // nullopt

    auto result = present.value_or(cc);
    EXPECT_EQ(typeid(std::string), typeid(result));
    EXPECT_EQ("present"s, result);

    result = absent.value_or(cc);
    EXPECT_EQ("true"s, result);

    // Check move semantics in argument:

    using nocopy = testing::nocopy<int>;

    nocopy::reset_counts();
    nocopy z1 = optional<nocopy>().value_or(nocopy(7));

    EXPECT_EQ(7, z1.value);
    EXPECT_EQ(1, nocopy::move_ctor_count);

    nocopy::reset_counts();
    nocopy z2 = optional<nocopy>(nocopy(3)).value_or(nocopy(7));

    EXPECT_EQ(3, z2.value);
    EXPECT_EQ(2, nocopy::move_ctor_count);
}

TEST(optional, ref_value_or) {
    double a = 2.0;
    double b = 3.0;

    optional<double&> x = a;
    double& ref1 = x.value_or(b);

    EXPECT_EQ(2., ref1);

    x = nullopt;
    double& ref2 = x.value_or(b);

    EXPECT_EQ(3., ref2);

    ref1 = 12.;
    ref2 = 13.;
    EXPECT_EQ(12., a);
    EXPECT_EQ(13., b);

    const optional<double&> cx = x;
    auto& ref3 = cx.value_or(b);
    EXPECT_TRUE(std::is_const<std::remove_reference_t<decltype(ref3)>>::value);
    EXPECT_EQ(&b, &ref3);
}

TEST(optional, void) {
    optional<void> a, b(true), c(a), d = b, e(false), f(nullopt);

    EXPECT_FALSE((bool)a);
    EXPECT_TRUE((bool)b);
    EXPECT_FALSE((bool)c);
    EXPECT_TRUE((bool)d);
    EXPECT_TRUE((bool)e);
    EXPECT_FALSE((bool)f);

    auto& check_rval = (b = nullopt);
    EXPECT_FALSE((bool)b);
    EXPECT_EQ(&b, &check_rval);
}

TEST(optional, conversion) {
    optional<double> a(3), b = 5;
    EXPECT_TRUE((bool)a);
    EXPECT_TRUE((bool)b);
    EXPECT_EQ(3.0, a.value());
    EXPECT_EQ(5.0, b.value());

    optional<int> x;
    optional<double> c(x);
    optional<double> d = optional<int>();
    EXPECT_FALSE((bool)c);
    EXPECT_FALSE((bool)d);
}

TEST(optional, just) {
    int x = 3;

    optional<int&> o1 = just(x);
    optional<int>  o2 = just(x);

    o1.value() = 4;
    optional<int>  o3 = just(x);
    EXPECT_EQ(4, o1.value());
    EXPECT_EQ(3, o2.value());
    EXPECT_EQ(4, o3.value());
}

TEST(optional, emplace) {
    optional<int> o1(7);
    optional<std::array<double, 3>> o2{{22., 22., 22.}};
    int x = 42;
    std::array<double, 3> arr{{4.5, 7.1, 1.2}};

    o1.emplace(x);
    o2.emplace(arr);

    EXPECT_EQ(42, o1.value());
    EXPECT_EQ(4.5, o2.value()[0]);
    EXPECT_EQ(7.1, o2.value()[1]);
    EXPECT_EQ(1.2, o2.value()[2]);
}
