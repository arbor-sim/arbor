#include <iostream>

#include <arbor/util/any_cast.hpp>
#include <arbor/util/unique_any.hpp>

#include "util/rangeutil.hpp"
#include "util/span.hpp"

#include <gtest/gtest.h>
#include "common.hpp"

using namespace arb;
using util::any_cast;

TEST(unique_any, copy_construction) {
    using util::unique_any;

    unique_any any_int(2);
    EXPECT_EQ(any_int.type(), typeid(int));

    unique_any any_float(2.0f);
    EXPECT_EQ(any_float.type(), typeid(float));

    std::string str = "hello";
    unique_any any_string(str);
    EXPECT_EQ(any_string.type(), typeid(std::string));
}

namespace {
    struct moveable {
        moveable() = default;

        moveable(moveable&& other):
            moves(other.moves+1), copies(other.copies)
        {}

        moveable(const moveable& other):
            moves(other.moves), copies(other.copies+1)
        {}

        int moves=0;
        int copies=0;
    };
}

TEST(unique_any, move_construction) {
    moveable m;

    util::unique_any copied(m);
    util::unique_any moved(std::move(m));

    // Check that the expected number of copies and moves were performed.
    const auto& cref = any_cast<const moveable&>(copied);
    EXPECT_EQ(cref.moves, 0);
    EXPECT_EQ(cref.copies, 1);

    const auto& mref = any_cast<const moveable&>(moved);
    EXPECT_EQ(mref.moves, 1);
    EXPECT_EQ(mref.copies, 0);

    // construction by any&& should not make any copies or moves of the
    // constructed value
    util::unique_any fin(std::move(moved));
    EXPECT_FALSE(moved.has_value()); // moved has been moved from and should be empty
    const auto& fref = any_cast<const moveable&>(fin);
    EXPECT_EQ(fref.moves, 1);
    EXPECT_EQ(fref.copies, 0);

    const auto value = any_cast<moveable>(fin);
    EXPECT_EQ(value.moves, 1);
    EXPECT_EQ(value.copies, 1);
}

TEST(unique_any, type) {
    using util::unique_any;

    unique_any anyi(42);
    unique_any anys(std::string("hello"));
    unique_any anyv(std::vector<int>{1, 2, 3});
    unique_any any0;

    EXPECT_EQ(typeid(int), anyi.type());
    EXPECT_EQ(typeid(std::string), anys.type());
    EXPECT_EQ(typeid(std::vector<int>), anyv.type());
    EXPECT_EQ(typeid(void), any0.type());

    anyi.reset();
    EXPECT_EQ(typeid(void), anyi.type());

    anyi = std::true_type();
    EXPECT_EQ(typeid(std::true_type), anyi.type());
}

TEST(unique_any, swap) {
    using util::unique_any;

    unique_any any1(42);     // integer
    unique_any any2(3.14);   // double

    EXPECT_EQ(typeid(int),    any1.type());
    EXPECT_EQ(typeid(double), any2.type());

    any1.swap(any2);

    EXPECT_EQ(any_cast<int>(any2), 42);
    EXPECT_EQ(any_cast<double>(any1), 3.14);

    EXPECT_EQ(typeid(double), any1.type());
    EXPECT_EQ(typeid(int),    any2.type());

    any1.swap(any2);

    EXPECT_EQ(any_cast<double>(any2), 3.14);
    EXPECT_EQ(any_cast<int>(any1), 42);

    EXPECT_EQ(typeid(int),    any1.type());
    EXPECT_EQ(typeid(double), any2.type());
}

TEST(unique_any, not_copy_constructable) {
    using T = testing::nocopy<int>;

    util::unique_any a(T(42));

    auto& ref = any_cast<T&>(a);
    EXPECT_EQ(ref, 42);
    ref.value = 100;

    EXPECT_EQ(any_cast<T&>(a).value, 100);

    // the following will fail if uncommented, because we are requesting
    // a copy of an non-copyable type.

    //auto value = any_cast<T>(a);

    // Test that we can move the contents of the unique_any.
    // NOTE: it makes sense to sink a with std::move(a) instead
    // of the following, because after such an assignment a will
    // be in a moved from state, and enforcing std::move(a) it
    // is made clearer in the calling code that a has been invalidated.
    //     any_cast<T&&>(a)
    T val(any_cast<T&&>(std::move(a)));
    EXPECT_EQ(val.value, 100);
}

// test any_cast(unique_any*) and any_cast(const unique_any*)
//   - these have different behavior to any_cast on reference types
//   - are used by the any_cast on reference types
TEST(unique_any, any_cast_ptr) {
    using util::unique_any;

    // test that valid pointers are returned for int and std::string types

    unique_any ai(42);
    auto ptr_i = any_cast<int>(&ai);
    EXPECT_EQ(*ptr_i, 42);

    unique_any as(std::string("hello"));
    auto ptr_s = any_cast<std::string>(&as);
    EXPECT_EQ(*ptr_s, "hello");

    // test that exceptions are thrown for invalid casts
    EXPECT_EQ(any_cast<int>(&as), nullptr);
    EXPECT_EQ(any_cast<std::string>(&ai), nullptr);
    unique_any empty;
    EXPECT_EQ(any_cast<int>(&empty), nullptr);
    EXPECT_EQ(any_cast<int>((util::unique_any*)nullptr), nullptr);

    // Check that constness of the returned pointer matches that the input.
    // Check that constness of the returned pointer matches that the input.
    {
        unique_any a(42);
        auto p = any_cast<int>(&a);

        // any_cast(any*) should not return const*
        EXPECT_TRUE((std::is_same<int*, decltype(p)>::value));
    }
    {
        const unique_any a(42);
        auto p = any_cast<int>(&a);

        // any_cast(const any*) should return const*
        EXPECT_TRUE((std::is_same<const int*, decltype(p)>::value));
    }
}

// test any_cast(unique_any&)
TEST(unique_any, any_cast_ref) {
    util::unique_any ai(42);
    auto& i = any_cast<int&>(ai);

    EXPECT_EQ(typeid(i), typeid(int));
    EXPECT_EQ(i, 42);

    // any_cast<T>(unique_any&) returns a 
    i = 100;
    EXPECT_EQ(any_cast<int>(ai), 100);
}

// test any_cast(const unique_any&)
TEST(unique_any, any_cast_const_ref) {
    const util::unique_any ai(42);
    auto& i = any_cast<const int&>(ai);

    EXPECT_EQ(typeid(i), typeid(int));
    EXPECT_EQ(i, 42);

    EXPECT_TRUE((std::is_same<const int&, decltype(i)>::value));
}

// test any_cast(unique_any&&)
TEST(unique_any, any_cast_rvalue) {
    auto moved = any_cast<moveable>(util::unique_any(moveable()));
    EXPECT_EQ(moved.moves, 2);
    EXPECT_EQ(moved.copies, 0);
}

TEST(unique_any, std_swap) {
    util::unique_any a1(42);
    util::unique_any a2(3.14);

    auto pi = any_cast<int>(&a1);
    auto pd = any_cast<double>(&a2);

    std::swap(a1, a2);

    // test that values were swapped
    EXPECT_EQ(any_cast<int>(a2), 42);
    EXPECT_EQ(any_cast<double>(a1), 3.14);

    // test that underlying pointers did not change
    EXPECT_EQ(pi, any_cast<int>(&a2));
    EXPECT_EQ(pd, any_cast<double>(&a1));
}

// test operator=(unique_any&&)
TEST(unique_any, assignment_from_rvalue) {
    using util::unique_any;
    using std::string;

    auto str1 = string("one");
    auto str2 = string("two");
    unique_any a(str1);

    unique_any b;
    b = std::move(a); // move assignment

    EXPECT_EQ(str1, any_cast<string>(b));

    EXPECT_EQ(nullptr, any_cast<string>(&a));
}

// test template<typename T> operator=(T&&)
TEST(unique_any, assignment_from_value) {
    using util::unique_any;

    {
        std::vector<int> tmp{1, 2, 3};

        // take a pointer to the orignal data to later verify
        // that the value was moved, and not copied.
        auto ptr = tmp.data();

        unique_any a;
        a = std::move(tmp);

        auto vec = any_cast<std::vector<int>>(&a);

        // ensure the value was moved
        EXPECT_EQ(ptr, vec->data());

        // ensure that the contents of the vector are unchanged
        std::vector<int> ref{1, 2, 3};
        EXPECT_EQ(ref, *vec);
    }
    {
        using T = testing::nocopy<int>;

        T tmp(3);

        T::reset_counts();
        unique_any a;
        a = std::move(tmp);

        // the move constructor is called when constructing the
        // contained object
        EXPECT_EQ(T::move_ctor_count, 1);
        EXPECT_EQ(T::move_assign_count, 0);

        T::reset_counts();
        unique_any b = std::move(a);

        // no move of the underlying type because the swap between
        // is swapping the pointers to contained objects, not the
        // objects themselves.
        EXPECT_EQ(T::move_ctor_count, 0);
        EXPECT_EQ(T::move_assign_count, 0);
    }
}

TEST(unique_any, make_unique_any) {
    using util::make_unique_any;

    {
        auto a = make_unique_any<int>(42);

        EXPECT_EQ(typeid(int), a.type());
        EXPECT_EQ(42, any_cast<int>(a));
    }

    // check casting
    {
        auto a = make_unique_any<double>(42u);

        EXPECT_EQ(typeid(double), a.type());
        EXPECT_EQ(42.0, any_cast<double>(a));
    }

    // check forwarding of parameters to constructor
    {
        // create a string from const char*
        auto as = make_unique_any<std::string>("hello");

        EXPECT_EQ(any_cast<std::string>(as), std::string("hello"));

        // test forwarding of 0 size parameter list
        struct X {
            int value;
            X(): value(42) {}
        };
        auto ai = make_unique_any<X>();
        EXPECT_EQ(any_cast<X>(ai).value, 42);

        // test forwarding of 2 size parameter list
        auto av = make_unique_any<std::vector<int>>(3, 2);
        EXPECT_EQ(any_cast<std::vector<int>&>(av), (std::vector<int>{2, 2, 2}));
    }

    // test that we make_unique_any correctly forwards rvalue arguments to the constructor
    // of the contained object.
    {
        std::vector<int> tmp{1, 2, 3};

        // take a pointer to the orignal data to later verify
        // that the value was moved, and not copied.
        auto ptr = tmp.data();

        auto a = make_unique_any<std::vector<int>>(std::move(tmp));

        auto vec = any_cast<std::vector<int>>(&a);

        // ensure the value was moved
        EXPECT_EQ(ptr, vec->data());

        // ensure that the contents of the vector are unchanged
        std::vector<int> ref{1, 2, 3};
        EXPECT_EQ(ref, *vec);
    }
}

// test that unique_any plays nicely with STL containers
TEST(unique_any, stdvector)
{
    using util::unique_any;

    // push_back
    {
        using T = testing::nocopy<std::string>;
        auto get = [](const unique_any& v) {return any_cast<const T&>(v).value;};

        std::vector<unique_any> vec;
        vec.push_back(T("h"));
        vec.push_back(T("e"));
        vec.push_back(T("l"));
        vec.push_back(T("l"));
        vec.push_back(T("o"));

        std::string s;
        for (auto& v: vec) s += get(v);
        EXPECT_EQ(s, "hello");

        s.clear();
        vec.erase(std::begin(vec)+1);
        vec.erase(std::begin(vec)+1);
        vec.erase(std::begin(vec)+1);
        for (auto& v: vec) s += get(v);
        EXPECT_EQ(s, "ho");
    }

    // sort
    {
        auto get = [](const unique_any& v) {return any_cast<int>(v);};
        int n = 10;
        std::vector<unique_any> vec;

        // fill the vector with values in descending order:
        //  [n-1, n-2, ..., 1, 0]
        for (auto i: util::make_span(0, n)) {
            vec.emplace_back(n-i-1);
        }
        // sort to ascending order
        util::sort_by(vec, get);

        // verify sort
        for (auto i: util::make_span(0, n)) {
            EXPECT_EQ(i, get(vec[i]));
        }
    }

    // std::reverse with non-copyable type
    {
        using T = testing::nocopy<int>;
        auto get = [](const unique_any& v) {return any_cast<const T&>(v).value;};
        int n = 10;
        std::vector<unique_any> vec;

        // fill the vector with values in descending order:
        //  [n-1, n-2, ..., 1, 0]
        for (auto i: util::make_span(0, n)) {
            vec.emplace_back(T(n-i-1));
        }

        // sort to ascending order by reversing the vector, which is sorted in
        // descending order.
        std::reverse(vec.begin(), vec.end());

        // verify sort
        for (auto i: util::make_span(0, n)) {
            EXPECT_EQ(i, get(vec[i]));
        }
    }
}
