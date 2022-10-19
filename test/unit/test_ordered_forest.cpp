#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "util/ordered_forest.hpp"

using arb::util::ordered_forest;

namespace {
template <typename T>
struct simple_allocator {
    using value_type = T;

    simple_allocator():
        n_alloc_(new std::size_t()),
        n_dealloc_(new std::size_t())
    {}

    simple_allocator(const simple_allocator&) noexcept = default;

    template <typename U>
    simple_allocator(const simple_allocator<U>& other) noexcept {
        n_alloc_ = other.n_alloc_;
        n_dealloc_ = other.n_dealloc_;
    }

    T* allocate(std::size_t n) {
        auto p = new T[n];
        ++*n_alloc_;
        return p;
    }

    void deallocate(T* p, std::size_t) {
        delete [] p;
        ++*n_dealloc_;
    }

    bool operator==(const simple_allocator& other) const {
        return other.n_alloc_ == n_alloc_ && other.n_dealloc_ == n_dealloc_;
    }

    bool operator!=(const simple_allocator& other) const {
        return !(*this==other);
    }

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::true_type;

    void reset_counts() {
        *n_alloc_ = 0;
        *n_dealloc_ = 0;
    }

    std::size_t n_alloc() const { return *n_alloc_; }
    std::size_t n_dealloc() const { return *n_dealloc_; }

    std::shared_ptr<std::size_t> n_alloc_, n_dealloc_;
};

template <typename T, typename A>
::testing::AssertionResult forest_invariant(const ordered_forest<T, A>& f) {
    // check parent-child confluence:

    for (auto i = f.root_begin(); i!=f.root_end(); ++i) {
        if (i.parent()) {
            return ::testing::AssertionFailure() << "root node " << *i << " has parent " << *i.parent();
        }
    }

    for (auto i = f.begin(); i!=f.end(); ++i) {
        auto cb = f.child_begin(i);
        auto ce = f.child_end(i);

        for (auto j = cb; j!=ce; ++j) {
            if (j.parent() != i) {
                auto failure = ::testing::AssertionFailure() << "child node " << *j << " of " << *i << " has parent ";
                return j.parent()? (failure << *j.parent()): (failure << "(null)");
            }
        }
    }
    return ::testing::AssertionSuccess();
}

} // anonymous namespace

TEST(ordered_forest, empty) {
    ordered_forest<int> f1;
    EXPECT_EQ(0u, f1.size());
    EXPECT_TRUE(f1.begin() == f1.end());

    simple_allocator<int> alloc;
    ASSERT_EQ(0u, alloc.n_alloc());
    ASSERT_EQ(0u, alloc.n_dealloc());

    ordered_forest<int, simple_allocator<int>> f2(alloc);
    EXPECT_EQ(0u, alloc.n_alloc());
    EXPECT_EQ(0u, alloc.n_dealloc());
}

TEST(ordered_forest, push) {
    simple_allocator<int> alloc;

    {
        ordered_forest<int, simple_allocator<int>> f(alloc);

        f.push_front(3);
        auto i2 = f.push_front(2);
        f.push_child(i2, 5);
        f.push_child(i2, 4);
        f.push_front(1);

        ASSERT_TRUE(forest_invariant(f));

        ASSERT_EQ(5u, f.size());
        EXPECT_EQ(10u, alloc.n_alloc()); // five nodes, five items.

        auto i = f.begin();
        ASSERT_TRUE(i);
        EXPECT_EQ(1, *i);
        EXPECT_FALSE(i.child());

        i = i.next();
        ASSERT_TRUE(i);
        auto j = i.child();
        ASSERT_TRUE(j);
        EXPECT_EQ(4, *j);
        j = j.next();
        ASSERT_TRUE(j);
        EXPECT_EQ(5, *j);
        EXPECT_FALSE(j.next());

        i = i.next();
        ASSERT_TRUE(i);
        EXPECT_EQ(3, *i);
        EXPECT_FALSE(i.child());
        EXPECT_FALSE(i.next());
    }

    EXPECT_EQ(alloc.n_dealloc(), alloc.n_alloc());
}

TEST(ordered_forest, insert) {
    simple_allocator<int> alloc;

    {
        ordered_forest<int, simple_allocator<int>> f(alloc);

        auto r = f.push_front(1);
        f.insert_after(r, 3);
        r = f.insert_after(r, 2);
        auto c = f.push_child(r, 4);
        f.insert_after(c, 6);
        f.insert_after(c, 5);

        ASSERT_TRUE(forest_invariant(f));

        ASSERT_EQ(6u, f.size());
        EXPECT_EQ(12u, alloc.n_alloc()); // six nodes, six items.

        auto i = f.begin();
        ASSERT_TRUE(i);
        EXPECT_EQ(1, *i);
        EXPECT_FALSE(i.child());

        i = i.next();
        ASSERT_TRUE(i);
        auto j = i.child();
        ASSERT_TRUE(j);
        EXPECT_EQ(4, *j);
        j = j.next();
        ASSERT_TRUE(j);
        EXPECT_EQ(5, *j);
        j = j.next();
        ASSERT_TRUE(j);
        EXPECT_EQ(6, *j);
        EXPECT_FALSE(j.next());

        i = i.next();
        ASSERT_TRUE(i);
        EXPECT_EQ(3, *i);
        EXPECT_FALSE(i.child());
        EXPECT_FALSE(i.next());
    }

    EXPECT_EQ(alloc.n_dealloc(), alloc.n_alloc());
}

TEST(ordered_forest, initializer_list) {
    ordered_forest<int> f = {1, {2, {4, 5, 6}}, 3};
    EXPECT_EQ(6u, f.size());

    ASSERT_TRUE(forest_invariant(f));

    auto i = f.begin();
    ASSERT_TRUE(i);
    EXPECT_EQ(1, *i);
    EXPECT_FALSE(i.child());

    i = i.next();
    ASSERT_TRUE(i);
    auto j = i.child();
    ASSERT_TRUE(j);
    EXPECT_EQ(4, *j);
    j = j.next();
    ASSERT_TRUE(j);
    EXPECT_EQ(5, *j);
    j = j.next();
    ASSERT_TRUE(j);
    EXPECT_EQ(6, *j);
    EXPECT_FALSE(j.next());

    i = i.next();
    ASSERT_TRUE(i);
    EXPECT_EQ(3, *i);
    EXPECT_FALSE(i.child());
    EXPECT_FALSE(i.next());
}

TEST(ordered_forest, equality) {
    using of = ordered_forest<int>;

    EXPECT_EQ(of{}, of{});
    EXPECT_NE(of{1},  of{});
    EXPECT_NE(of{},  of{1});

    EXPECT_EQ((of{1, 2, 3}), (of{1, 2, 3}));
    EXPECT_NE((of{1, 2, 3}),  (of{1, {2, {3}}}));
    EXPECT_NE((of{{1, {2, 3}}}),  (of{1, 2, 3}));

    struct always_eq {
        int n_;
        always_eq(int n): n_(n) {}
        bool operator==(const always_eq&) const { return true; }
        bool operator!=(const always_eq&) const { return false; }
    };

    ordered_forest<always_eq> f1 = {{1, {2, 3}}, {4, {5, {6, {7}}, 8}}, 9};
    ordered_forest<always_eq> f2 = {{3, {1, 0}}, {2, {8, {6, {4}}, 7}}, 5};
    EXPECT_EQ(f2, f1);

    ordered_forest<always_eq> f3 = {{3, {{1, {0}}}}, {2, {8, {6, {4}}, 7}}, 5};
    EXPECT_NE(f1, f3);
}

TEST(ordered_forest, iteration) {
    using ivector = std::vector<int>;

    ordered_forest<int> f = {{1, {2, 3}}, {4, {5, {6, {7}}, 8}}, 9};
    const ordered_forest<int>& cf = f;

    ivector pre{f.preorder_begin(), f.preorder_end()};
    EXPECT_EQ((ivector{1, 2, 3, 4, 5, 6, 7, 8, 9}), pre);

    ivector cpre{cf.preorder_begin(), cf.preorder_end()};
    EXPECT_EQ((ivector{1, 2, 3, 4, 5, 6, 7, 8, 9}), cpre);

    ivector pre_default{f.begin(), f.end()};
    EXPECT_EQ((ivector{1, 2, 3, 4, 5, 6, 7, 8, 9}), pre_default);

    ivector cpre_default{cf.begin(), cf.end()};
    EXPECT_EQ((ivector{1, 2, 3, 4, 5, 6, 7, 8, 9}), cpre_default);

    ivector pre_cdefault{f.cbegin(), f.cend()};
    EXPECT_EQ((ivector{1, 2, 3, 4, 5, 6, 7, 8, 9}), pre_cdefault);

    ivector root{f.root_begin(), f.root_end()};
    EXPECT_EQ((ivector{1, 4, 9}), root);

    ivector croot{cf.root_begin(), cf.root_end()};
    EXPECT_EQ((ivector{1, 4, 9}), croot);

    ivector post{f.postorder_begin(), f.postorder_end()};
    EXPECT_EQ((ivector{2, 3, 1, 5, 7, 6, 8, 4, 9}), post);

    ivector cpost{cf.postorder_begin(), cf.postorder_end()};
    EXPECT_EQ((ivector{2, 3, 1, 5, 7, 6, 8, 4, 9}), cpost);

    auto four = std::find(f.begin(), f.end(), 4);
    ivector child_four{f.child_begin(four), f.child_end(four)};
    EXPECT_EQ((ivector{5, 6, 8}), child_four);

    ivector cchild_four{cf.child_begin(four), cf.child_end(four)};
    EXPECT_EQ((ivector{5, 6, 8}), cchild_four);

    using preorder_iterator = ordered_forest<int>::preorder_iterator;
    using postorder_iterator = ordered_forest<int>::postorder_iterator;

    ivector pre_four{preorder_iterator(four), preorder_iterator(four.next())};
    EXPECT_EQ((ivector{4, 5, 6, 7, 8}), pre_four);

    auto seven = std::find(f.begin(), f.end(), 7);
    auto nine = std::find(f.begin(), f.end(), 9);
    ivector post_seven_nine{postorder_iterator(seven), postorder_iterator(nine)};
    EXPECT_EQ((ivector{7, 6, 8, 4}), post_seven_nine);
}

TEST(ordered_forest, copy_move) {
    simple_allocator<int> alloc;

    using ivector = std::vector<int>;
    using of = ordered_forest<int, simple_allocator<int>>;

    of f1(alloc);
    {
        of f({{1, {2, 3}}, {4, {5, {6, {7}}, 8}}, 9}, alloc);
        EXPECT_EQ(18u, alloc.n_alloc());
        ASSERT_TRUE(forest_invariant(f));

        f1 = f;
        ASSERT_TRUE(forest_invariant(f1));
        EXPECT_EQ(36u, alloc.n_alloc());
        EXPECT_FALSE(f.empty());

        of f2 = std::move(f);
        ASSERT_TRUE(forest_invariant(f2));
        EXPECT_EQ(36u, alloc.n_alloc());
        EXPECT_TRUE(f.empty());

        ivector elems2{f2.begin(), f2.end()};
        EXPECT_EQ((ivector{1, 2, 3, 4, 5, 6, 7, 8, 9}), elems2);
    }

    EXPECT_EQ(36u, alloc.n_alloc());
    EXPECT_EQ(18u, alloc.n_dealloc());

    ivector elems1{f1.begin(), f1.end()};
    EXPECT_EQ((ivector{1, 2, 3, 4, 5, 6, 7, 8, 9}), elems1);

    // With a different != allocator object, require move assignment
    // to perform copy.

    simple_allocator<int> other_alloc;
    ASSERT_NE(alloc, other_alloc);
    ASSERT_FALSE(std::allocator_traits<simple_allocator<int>>::propagate_on_container_move_assignment::value);

    of f3(other_alloc);

    f3 = std::move(f1);
    ASSERT_TRUE(forest_invariant(f3));
    EXPECT_FALSE(f1.empty());

    EXPECT_EQ(36u, alloc.n_alloc());
    EXPECT_EQ(18u, alloc.n_dealloc());
    EXPECT_EQ(18u, other_alloc.n_alloc());
    EXPECT_EQ(0u, other_alloc.n_dealloc());
}

TEST(ordered_forest, erase) {
    simple_allocator<int> alloc;
    using of = ordered_forest<int, simple_allocator<int>>;

    of f({1, 2, {3, {4, {5, {6, 7}}, 8}}, 9}, alloc);
    ASSERT_TRUE(forest_invariant(f));
    EXPECT_EQ(18u, alloc.n_alloc());

    auto two = std::find(f.begin(), f.end(), 2);
    f.erase_after(two);

    ASSERT_TRUE(forest_invariant(f));
    EXPECT_EQ((of{1, 2, 4, {5, {6, 7}}, 8, 9}), f);
    EXPECT_EQ(2u, alloc.n_dealloc());

    auto five = std::find(f.begin(), f.end(), 5);
    f.erase_child(five);

    ASSERT_TRUE(forest_invariant(f));
    EXPECT_EQ((of{1, 2, 4, {5, {7}}, 8, 9}), f);
    EXPECT_EQ(4u, alloc.n_dealloc());

    auto eight = std::find(f.begin(), f.end(), 8);
    ASSERT_THROW(f.erase_child(eight), std::invalid_argument);

    auto seven = std::find(f.begin(), f.end(), 7);
    ASSERT_THROW(f.erase_after(seven), std::invalid_argument);

    f.erase_front();
    ASSERT_TRUE(forest_invariant(f));
    EXPECT_EQ((of{2, 4, {5, {7}}, 8, 9}), f);
    EXPECT_EQ(6u, alloc.n_dealloc());

    of empty;
    ASSERT_THROW(empty.erase_front(), std::invalid_argument);
}

TEST(ordered_forest, prune) {
    simple_allocator<int> alloc;
    using of = ordered_forest<int, simple_allocator<int>>;

    of f({1, 2, {3, {4, {5, {6, 7}}, 8}}, 9}, alloc);
    ASSERT_TRUE(forest_invariant(f));
    EXPECT_EQ(18u, alloc.n_alloc());
    EXPECT_EQ(0u, alloc.n_dealloc());

    of p1 = f.prune_front();
    ASSERT_TRUE(forest_invariant(f));
    ASSERT_TRUE(forest_invariant(p1));
    EXPECT_EQ((of{2, {3, {4, {5, {6, 7}}, 8}}, 9}), f);
    EXPECT_EQ((of{1}), p1);

    of p2 = f.prune_after(std::find(f.begin(), f.end(), 4));
    ASSERT_TRUE(forest_invariant(f));
    ASSERT_TRUE(forest_invariant(p2));
    EXPECT_EQ((of{2, {3, {4, 8}}, 9}), f);
    EXPECT_EQ((of{{5, {6, 7}}}), p2);

    of p3 = f.prune_child(std::find(f.begin(), f.end(), 3));
    ASSERT_TRUE(forest_invariant(f));
    ASSERT_TRUE(forest_invariant(p3));
    EXPECT_EQ((of{2, {3, {8}}, 9}), f);
    EXPECT_EQ((of{4}), p3);

    EXPECT_EQ(0u, alloc.n_dealloc());
    EXPECT_EQ(f.get_allocator(), p1.get_allocator());
    EXPECT_EQ(f.get_allocator(), p2.get_allocator());
    EXPECT_EQ(f.get_allocator(), p3.get_allocator());

    of empty;
    ASSERT_THROW(empty.erase_child(empty.begin()), std::invalid_argument);
    ASSERT_THROW(empty.erase_after(empty.begin()), std::invalid_argument);
    ASSERT_THROW(empty.erase_front(), std::invalid_argument);

    of unit{1};
    ASSERT_THROW(unit.erase_child(unit.begin()), std::invalid_argument);
    ASSERT_THROW(unit.erase_after(unit.begin()), std::invalid_argument);
}

TEST(ordered_forest, graft) {
    using of = ordered_forest<int, simple_allocator<int>>;

    of f1{1, {2, {3, 4}}, 5};
    auto j = f1.graft_after(f1.begin(), of{6, {7, {8}}});

    ASSERT_TRUE(j);
    ASSERT_TRUE(forest_invariant(f1));
    EXPECT_EQ(7, *j);
    EXPECT_EQ((of{1, 6, {7, {8}}, {2, {3, 4}}, 5}), f1);

    j = f1.graft_child(std::find(f1.begin(), f1.end(), 2), of{9, 10});

    ASSERT_TRUE(j);
    ASSERT_TRUE(forest_invariant(f1));
    EXPECT_EQ(10, *j);
    EXPECT_EQ((of{1, 6, {7, {8}}, {2, {9, 10, 3, 4}}, 5}), f1);

    j = f1.graft_front(of{{11, {12, 13}}});

    ASSERT_TRUE(j);
    ASSERT_TRUE(forest_invariant(f1));
    EXPECT_EQ(11, *j);
    EXPECT_EQ((of{{11, {12, 13}}, 1, 6, {7, {8}}, {2, {9, 10, 3, 4}}, 5}), f1);

    simple_allocator<int> alloc1, alloc2;
    of f2({1, 2}, alloc1);
    of f3({3, 4}, alloc1);
    of f4({5, 6}, alloc2);

    ASSERT_TRUE(forest_invariant(f2));
    ASSERT_TRUE(forest_invariant(f3));
    ASSERT_TRUE(forest_invariant(f4));

    EXPECT_EQ(8u, alloc1.n_alloc());
    EXPECT_EQ(0u, alloc1.n_dealloc());
    EXPECT_EQ(4u, alloc2.n_alloc());

    f2.graft_front(std::move(f3));
    ASSERT_TRUE(forest_invariant(f2));
    ASSERT_TRUE(forest_invariant(f3));
    EXPECT_EQ(8u, alloc1.n_alloc());
    EXPECT_EQ(0u, alloc1.n_dealloc());

    f2.graft_front(std::move(f4));
    ASSERT_TRUE(forest_invariant(f2));
    ASSERT_TRUE(forest_invariant(f4));
    EXPECT_EQ(12u, alloc1.n_alloc());
    EXPECT_EQ(0u, alloc1.n_dealloc());
    EXPECT_EQ(4u, alloc2.n_dealloc());

    EXPECT_EQ((of{5, 6, 3, 4, 1, 2}), f2);
}

TEST(ordered_forest, swap) {
    simple_allocator<int> alloc1, alloc2;
    using of = ordered_forest<int, simple_allocator<int>>;

    of a({1, {2, {3, 4}}, 5}, alloc1);
    of b({6, 7, 8, 9}, alloc2);

    of a_copy(a), b_copy(b);

    ASSERT_TRUE(forest_invariant(a));
    ASSERT_TRUE(forest_invariant(b));
    ASSERT_TRUE(forest_invariant(a_copy));
    ASSERT_TRUE(forest_invariant(b_copy));

    ASSERT_EQ(alloc1, a.get_allocator());
    ASSERT_EQ(alloc2, b.get_allocator());
    ASSERT_EQ(alloc1, a_copy.get_allocator());
    ASSERT_EQ(alloc2, b_copy.get_allocator());
    ASSERT_EQ(a_copy, a);
    ASSERT_EQ(b_copy, b);

    swap(a, b);

    ASSERT_TRUE(forest_invariant(a));
    ASSERT_TRUE(forest_invariant(b));
    EXPECT_EQ(alloc2, a.get_allocator());
    EXPECT_EQ(alloc1, b.get_allocator());
    EXPECT_EQ(b_copy, a);
    EXPECT_EQ(a_copy, b);
}
