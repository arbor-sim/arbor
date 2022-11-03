#include <any>
#include <string>
#include <type_traits>
#include <typeinfo>

#include <arbor/util/any_visitor.hpp>

#include <gtest/gtest.h>
#include "common.hpp"

using namespace std::string_literals;
using std::any;
using std::any_cast;
using std::bad_any_cast;

using arb::util::overload;
using arb::util::any_visitor;

TEST(any_visitor, simple) {
    enum { A0, B0, C0 };
    struct A { int value = A0; };
    struct B { int value = B0; };
    struct C { int value = C0; };

    using V = any_visitor<A, B, C>;

    auto get_value = [](auto y) { return y.value; };

    EXPECT_EQ(A0, V::visit(get_value, any(A{})));
    EXPECT_EQ(B0, V::visit(get_value, any(B{})));
    EXPECT_EQ(C0, V::visit(get_value, any(C{})));
}

TEST(any_visitor, heterogeneous) {
    struct r_base { int value; };
    struct r_derived: r_base { int other; };

    struct D { r_base item; };
    struct E { r_derived item; };

    using DE_visitor = any_visitor<D, E>;
    using E_visitor = any_visitor<E>;

    auto get_item = [](auto y) { return y.item; };

    // Return type is common type across possible returns of visiting functor.

    {
        auto result = E_visitor::visit(get_item, any(E{}));
        EXPECT_TRUE((std::is_same<decltype(result), r_derived>::value));
    }
    {
        auto result = DE_visitor::visit(get_item, any(E{}));
        EXPECT_TRUE((std::is_same<decltype(result), r_base>::value));
    }
}

TEST(any_visitor, unmatched) {
    enum { A0, B0, C0, D0 };
    struct A { int value = A0; };
    struct B { int value = B0; };
    struct C { int value = C0; };
    struct D { int value = D0; };

    using V = any_visitor<A, B, C>;

    struct catch_unmatched {
        bool operator()(A) const { return true; }
        bool operator()(B) const { return true; }
        bool operator()(C) const { return true; }
        bool operator()() const { return false; }
    } f;

    EXPECT_EQ(true,  V::visit(f, any(A{})));
    EXPECT_EQ(true,  V::visit(f, any(B{})));
    EXPECT_EQ(true,  V::visit(f, any(C{})));
    EXPECT_EQ(false, V::visit(f, any(D{})));

    auto get_value = [](auto y) { return y.value; };

    EXPECT_NO_THROW(V::visit(get_value, any(A{})));
    EXPECT_NO_THROW(V::visit(get_value, any(A{})));
    EXPECT_NO_THROW(V::visit(get_value, any(A{})));
    EXPECT_THROW(V::visit(get_value, any(D{})), bad_any_cast);
}

TEST(any_visitor, preserve_visiting_ref) {
    struct check_ref {
        check_ref(int& result): result(result) {}
        int& result;

        void operator()(...) & { result = 1; };
        void operator()(...) && { result = 2; };
    };

    struct A {};
    struct B {};

    int result = 0;
    check_ref lref(result);

    any_visitor<A, B>::visit(lref, any(B{}));
    EXPECT_EQ(1, result);

    result = 0;
    any_visitor<A, B>::visit(check_ref(result), any(B{}));
    EXPECT_EQ(2, result);
}

TEST(any_visitor, preserve_any_qual) {
    struct A { int x = 0; };
    struct B { int x = 0; };

    struct check_qual {
       int operator()(const A&) { return 0; }
       int operator()(A&) { return 1; }
       int operator()(A&&) { return 2; }

       int operator()(const B&) { return 3; }
       int operator()(B&) { return 4; }
       int operator()(B&&) { return 5; }
    };

    using V = any_visitor<A, B>;
    check_qual q;

    any a(A{});
    const any const_a(A{});
    EXPECT_EQ(0, V::visit(q, const_a));
    EXPECT_EQ(1, V::visit(q, a));
    EXPECT_EQ(2, V::visit(q, any(A{})));

    any b(B{});
    const any const_b(B{});
    EXPECT_EQ(3, V::visit(q, const_b));
    EXPECT_EQ(4, V::visit(q, b));
    EXPECT_EQ(5, V::visit(q, any(B{})));

    any v(B{10});
    V::visit([](auto& u) { ++u.x; }, v);
    EXPECT_EQ(11, any_cast<B>(v).x);
}

TEST(overload, simple) {
    struct A { int value; };
    struct B { double value; };

    auto f = overload([](A a) { return a.value+10; },
                      [](B b) { return b.value+20; });

    EXPECT_TRUE((std::is_same<decltype(f(A{})), int>::value));
    EXPECT_TRUE((std::is_same<decltype(f(B{})), double>::value));

    EXPECT_EQ(12,   f(A{2}));
    EXPECT_EQ(22.5, f(B{2.5}));
}

namespace {
int f1(int) { return 1; }
int f2(int) { return 2; }
}

TEST(overload, precedence) {
    // Overloaded functional object will match the first invocable operator(), not the best match.

    EXPECT_EQ(1, overload(f1, f2)(0));
    EXPECT_EQ(2, overload(f2, f1)(0));

    EXPECT_EQ(1, overload([](int) { return 1; }, [](double) { return 2; })(0));
    EXPECT_EQ(1, overload([](int) { return 1; }, [](double) { return 2; })(2.3));
}

TEST(overload, qualified_match) {
    struct A {};

    auto f = overload(
        [](A&&) { return 1; },
        [](const A&&) { return 2; },
        [](A&) { return 3; },
        [](const A&) { return 4; });

    A a;
    const A const_a;

    EXPECT_EQ(1, f(std::move(a)));
    EXPECT_EQ(2, f(std::move(const_a)));
    EXPECT_EQ(3, f(a));
    EXPECT_EQ(4, f(const_a));
}


