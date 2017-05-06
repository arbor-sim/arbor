#include "../gtest.h"
#include "common.hpp"

#include <iostream>

#include <util/any.hpp>

using namespace nest::mc;

TEST(any, copy_construction) {
    util::any any_int(2);
    EXPECT_EQ(any_int.type(), typeid(int));

    util::any any_float(2.0f);
    EXPECT_EQ(any_float.type(), typeid(float));

    std::string str = "hello";
    util::any any_string(str);
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

TEST(any, move_construction) {
    moveable m;

    util::any copied(m);
    util::any moved(std::move(m));

    // Check that the expected number of copies and moves were performed.
    // Note that any_cast(any*) is used instead of any_cast(const any&) because
    // any_cast(const any&) returns a copy.
    const auto& cref = *util::any_cast<moveable>(&copied);
    EXPECT_EQ(cref.moves, 0);
    EXPECT_EQ(cref.copies, 1);

    const auto& mref = *util::any_cast<moveable>(&moved);
    EXPECT_EQ(mref.moves, 1);
    EXPECT_EQ(mref.copies, 0);

    // construction by any&& should not make any copies or moves of the
    // constructed value
    util::any fin(std::move(moved));
    EXPECT_FALSE(moved.has_value()); // moved has been moved from and should be empty
    const auto& fref = *util::any_cast<moveable>(&fin);
    EXPECT_EQ(fref.moves, 1);
    EXPECT_EQ(fref.copies, 0);

    const auto value = util::any_cast<moveable>(fin);
    EXPECT_EQ(value.moves, 1);
    EXPECT_EQ(value.copies, 1);
}

TEST(any, type) {
    using util::any;

    any anyi(42);
    any anys(std::string("hello"));
    any anyv(std::vector<int>{1, 2, 3});
    any any0;

    EXPECT_EQ(typeid(int), anyi.type());
    EXPECT_EQ(typeid(std::string), anys.type());
    EXPECT_EQ(typeid(std::vector<int>), anyv.type());
    EXPECT_EQ(typeid(void), any0.type());

    anyi.reset();
    EXPECT_EQ(typeid(void), anyi.type());

    anyi = std::true_type();
    EXPECT_EQ(typeid(std::true_type), anyi.type());
}

TEST(any, swap) {
    using util::any;
    using util::any_cast;

    any any1(42);     // integer
    any any2(3.14);   // double

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

TEST(any, constness) {
}

// These should fail at compile time if the constraint that the contents of any
// satisfy CopyConstructable. This implementation is rock solid, so they have
// to be commented out.
/*
TEST(any, not_copy_constructable) {
    util::any a(testing::nocopy<int>(3));

    testing::nocopy<int> x(3);
    util::any b(std::move(x));
}
*/

// test any_cast(any*)
//   - these have different behavior to any_cast on reference types
//   - are used by the any_cast on refernce types
TEST(any, any_cast_ptr) {

    // test that valid pointers are returned for int and std::string types

    util::any ai(42);
    auto ptr_i = util::any_cast<int>(&ai);
    EXPECT_EQ(*ptr_i, 42);

    util::any as(std::string("hello"));
    auto ptr_s = util::any_cast<std::string>(&as);
    EXPECT_EQ(*ptr_s, "hello");

    // test that exceptions are thrown for invalid casts
    EXPECT_EQ(util::any_cast<int>(&as), nullptr);
    EXPECT_EQ(util::any_cast<std::string>(&ai), nullptr);
    util::any empty;
    EXPECT_EQ(util::any_cast<int>(&empty), nullptr);
    EXPECT_EQ(util::any_cast<int>((util::any*)nullptr), nullptr);

    // Check that constness of the returned pointer matches that the input.
    {
        util::any a(42);
        auto p = util::any_cast<int>(&a);
        static_assert(std::is_same<int*, decltype(p)>::value,
                "any_cast(any*) should not return const*");
    }
    {
        const util::any a(42);
        auto p = util::any_cast<int>(&a);
        static_assert(std::is_same<const int*, decltype(p)>::value,
                "any_cast(const any*) should return const*");
    }
}

TEST(any, any_cast_ref) {
    util::any ai(42);
    auto i = util::any_cast<int>(ai);
    EXPECT_EQ(typeid(i), typeid(int));
    EXPECT_EQ(i, 42);
}

// test any_cast(any&&)
TEST(any, any_cast_rvalue) {
    auto moved = util::any_cast<moveable>(util::any(moveable()));
    EXPECT_EQ(moved.moves, 2);
    EXPECT_EQ(moved.copies, 0);
}

TEST(any, std_swap) {
    util::any a1(42);
    util::any a2(3.14);

    auto pi = util::any_cast<int>(&a1);
    auto pd = util::any_cast<double>(&a2);

    std::swap(a1, a2);

    // test that values were swapped
    EXPECT_EQ(util::any_cast<int>(a2), 42);
    EXPECT_EQ(util::any_cast<double>(a1), 3.14);

    // test that underlying pointers did not change
    EXPECT_EQ(pi, util::any_cast<int>(&a2));
    EXPECT_EQ(pd, util::any_cast<double>(&a1));
}

// test operator=(const any&)
TEST(any, assignment_from_lvalue) {
    using std::string;

    auto str1 = string("one");
    auto str2 = string("two");
    util::any a(str1);

    util::any b;
    b = a; // copy assignment

    // verify that b contains value stored in a
    EXPECT_EQ(str1, util::any_cast<string>(b));

    // change the value stored in b
    *util::any_cast<string>(&b) = str2;

    // verify that a is unchanged and that b holds new value
    EXPECT_EQ(str1, util::any_cast<string>(a));
    EXPECT_EQ(str2, util::any_cast<string>(b));
}

// test operator=(any&&)
TEST(any, assignment_from_rvalue) {
    using std::string;

    auto str1 = string("one");
    auto str2 = string("two");
    util::any a(str1);

    util::any b;
    b = std::move(a); // move assignment

    EXPECT_EQ(str1, util::any_cast<string>(b));

    EXPECT_EQ(nullptr, util::any_cast<string>(&a));
}

// test template<typename T> operator=(T&&)
TEST(any, assignment_from_value) {
    std::vector<int> tmp{1, 2, 3};

    // take a pointer to the orignal data to later verify
    // that the value was moved, and not copied.
    auto ptr = tmp.data();

    util::any a;
    a = std::move(tmp);

    auto vec = util::any_cast<std::vector<int>>(&a);

    // ensure the value was moved
    EXPECT_EQ(ptr, vec->data());

    // ensure that the vector was copied correctly
    std::vector<int> ref{1, 2, 3};
    EXPECT_EQ(ref, *vec);
}
