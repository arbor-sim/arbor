#include "../gtest.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <list>
#include <numeric>
#include <type_traits>

#ifdef NMC_HAVE_TBB
#include <tbb/tbb_stddef.h>
#endif

#include <util/counter.hpp>
#include <util/meta.hpp>
#include <util/range.hpp>
#include <util/rangeutil.hpp>
#include <util/sentinel.hpp>
#include <util/transform.hpp>

#include "common.hpp"

using namespace nest::mc;
using testing::null_terminated;

TEST(range, list_iterator) {
    std::list<int> l = { 2, 4, 6, 8, 10 };

    auto s  = util::make_range(l.begin(), l.end());

    EXPECT_EQ(s.left, l.begin());
    EXPECT_EQ(s.right, l.end());

    EXPECT_EQ(s.begin(), l.begin());
    EXPECT_EQ(s.end(), l.end());

    EXPECT_EQ(s.size(), l.size());
    EXPECT_EQ(s.front(), *l.begin());
    EXPECT_EQ(s.back(), *std::prev(l.end()));

    int check = std::accumulate(l.begin(), l.end(), 0);
    int sum = 0;
    for (auto i: s) {
        sum += i;
    }

    EXPECT_EQ(check, sum);

    auto sum2 = std::accumulate(s.begin(), s.end(), 0);
    EXPECT_EQ(check, sum2);

    // Check that different begin/end iterators are treated correctly
    auto sc = util::make_range(l.begin(), l.cend());
    EXPECT_EQ(l.size(), sc.size());
}

TEST(range, pointer) {
    int xs[] = { 10, 11, 12, 13, 14, 15, 16 };
    int l = 2;
    int r = 5;

    util::range<int *> s(&xs[l], &xs[r]);
    auto s_deduced = util::make_range(xs+l, xs+r);

    EXPECT_TRUE((std::is_same<decltype(s), decltype(s_deduced)>::value));
    EXPECT_EQ(s.left, s_deduced.left);
    EXPECT_EQ(s.right, s_deduced.right);

    EXPECT_EQ(3u, s.size());

    EXPECT_EQ(xs[l], *s.left);
    EXPECT_EQ(xs[l], *s.begin());
    EXPECT_EQ(xs[l], s[0]);
    EXPECT_EQ(xs[l], s.front());

    EXPECT_EQ(xs[r], *s.right);
    EXPECT_EQ(xs[r], *s.end());
    EXPECT_THROW(s.at(r-l), std::out_of_range);

    EXPECT_EQ(r-l, std::distance(s.begin(), s.end()));

    EXPECT_TRUE(std::equal(s.begin(), s.end(), &xs[l]));
}

TEST(range, empty) {
    int xs[] = { 10, 11, 12, 13, 14, 15, 16 };
    auto l = 2;
    auto r = 5;

    auto empty_range_ll = util::make_range((const int *) &xs[l], &xs[l]);
    EXPECT_TRUE(empty_range_ll.empty());
    EXPECT_EQ(empty_range_ll.begin() == empty_range_ll.end(),
              empty_range_ll.empty());
    EXPECT_EQ(0u, empty_range_ll.size());


    auto empty_range_rr = util::make_range(&xs[r], (const int *) &xs[r]);
    EXPECT_TRUE(empty_range_rr.empty());
    EXPECT_EQ(empty_range_rr.begin() == empty_range_rr.end(),
              empty_range_rr.empty());
    EXPECT_EQ(0u, empty_range_rr.size());
}

// This is the same test for enabling the incrementing util::distance
// implementation.
template <typename I, typename E>
constexpr bool uses_incrementing_distance() {
    return !util::has_common_random_access_iterator<I, E>::value &&
        util::is_forward_iterator<I>::value;
}

TEST(range, size_implementation) {
    ASSERT_TRUE((uses_incrementing_distance<
                     std::list<int>::iterator,
                      std::list<int>::const_iterator
                >()));

    ASSERT_FALSE((uses_incrementing_distance<
                      std::vector<int>::iterator,
                      std::vector<int>::const_iterator
                >()));

    ASSERT_FALSE((uses_incrementing_distance<int*, int*>()));

    ASSERT_FALSE((uses_incrementing_distance<const int*, int*>()));

    ASSERT_FALSE((uses_incrementing_distance<int*, const int*>()));
}

TEST(range, input_iterator) {
    int nums[] = { 10, 9, 8, 7, 6 };
    std::istringstream sin("10 9 8 7 6");
    auto s = util::make_range(std::istream_iterator<int>(sin),
                              std::istream_iterator<int>());

    EXPECT_TRUE(std::equal(s.begin(), s.end(), &nums[0]));
}

TEST(range, const_iterator) {
    std::vector<int> xs = { 1, 2, 3, 4, 5 };
    auto r = util::make_range(xs.begin(), xs.end());
    EXPECT_TRUE((std::is_same<int&, decltype(r.front())>::value));

    const auto& xs_const = xs;
    auto r_const = util::make_range(xs_const.begin(), xs_const.end());
    EXPECT_TRUE((std::is_same<const int&, decltype(r_const.front())>::value));
}

TEST(range, view) {
    std::vector<int> xs = { 1, 2, 3, 4, 5 };
    auto r = util::range_view(xs);

    r[3] = 7;
    std::vector<int> check = { 1, 2, 3, 7, 5 };
    EXPECT_EQ(check, xs);
}

TEST(range, sentinel) {
    const char *cstr = "hello world";
    std::string s;

    auto cstr_range = util::make_range(cstr, null_terminated);
    for (auto i=cstr_range.begin(); i!=cstr_range.end(); ++i) {
        s += *i;
    }

    EXPECT_EQ(s, std::string(cstr));
    EXPECT_EQ(s.size(), cstr_range.size());

    s.clear();
    for (auto c: canonical_view(cstr_range)) {
        s += c;
    }

    EXPECT_EQ(s, std::string(cstr));

    const char *empty_cstr = "";
    auto empty_cstr_range = util::make_range(empty_cstr, null_terminated);
    EXPECT_TRUE(empty_cstr_range.empty());
    EXPECT_EQ(0u, empty_cstr_range.size());
}

TEST(range, strictify) {
    const char *cstr = "hello world";
    auto cstr_range = util::make_range(cstr, null_terminated);

    auto ptr_range = util::strict_view(cstr_range);
    EXPECT_TRUE((std::is_same<decltype(ptr_range), util::range<const char *>>::value));
    EXPECT_EQ(cstr, ptr_range.left);
    EXPECT_EQ(cstr+11, ptr_range.right);
}

TEST(range, max_element_by) {
    const char *cstr = "helloworld";
    auto cstr_range = util::make_range(cstr, null_terminated);

    auto i = util::max_element_by(cstr_range,
        [](char c) -> int { return -c; });

    EXPECT_TRUE((std::is_same<const char *, decltype(i)>::value));
    EXPECT_EQ('d', *i);
    EXPECT_EQ(cstr+9, i);

    // with mutable container
    std::vector<int> v = { 1, 3, -5, 2 };
    auto j  = util::max_element_by(v, [](int x) { return x*x; });
    EXPECT_EQ(-5, *j);
    *j = 1;
    j  = util::max_element_by(v, [](int x) { return x*x; });
    EXPECT_EQ(3, *j);
}

TEST(range, max_value) {
    const char *cstr = "hello world";
    auto cstr_range = util::make_range(cstr, null_terminated);

    // use a lambda to get a range over non-assignable iterators
    // (at least until we specialize `transform_iterator` for
    // non-copyable functors passed by const reference).
    auto i = util::max_value(
        util::transform_view(cstr_range, [](char c) { return c+1; }));

    EXPECT_EQ('x', i);
}


template <typename V>
class counter_range: public ::testing::Test {};

TYPED_TEST_CASE_P(counter_range);

TYPED_TEST_P(counter_range, max_size) {
    using int_type = TypeParam;
    using unsigned_int_type = typename std::make_unsigned<int_type>::type;
    using counter = util::counter<int_type>;

    auto l = counter{int_type{1}};
    auto r = counter{int_type{10}};
    auto range = util::make_range(l, r);
    EXPECT_EQ(std::numeric_limits<unsigned_int_type>::max(),
              range.max_size());

}

TYPED_TEST_P(counter_range, extreme_size) {
    using int_type = TypeParam;
    using signed_int_type = typename std::make_signed<int_type>::type;
    using unsigned_int_type = typename std::make_unsigned<int_type>::type;
    using counter = util::counter<signed_int_type>;

    auto l = counter{std::numeric_limits<signed_int_type>::min()};
    auto r = counter{std::numeric_limits<signed_int_type>::max()};
    auto range = util::make_range(l, r);
    EXPECT_FALSE(range.empty());
    EXPECT_EQ(std::numeric_limits<unsigned_int_type>::max(), range.size());
}


TYPED_TEST_P(counter_range, size) {
    using int_type = TypeParam;
    using signed_int_type = typename std::make_signed<int_type>::type;
    using counter = util::counter<signed_int_type>;

    auto l = counter{signed_int_type{-3}};
    auto r = counter{signed_int_type{3}};
    auto range = util::make_range(l, r);
    EXPECT_FALSE(range.empty());
    EXPECT_EQ(6u, range.size());
}

TYPED_TEST_P(counter_range, at) {
    using int_type = TypeParam;
    using signed_int_type = typename std::make_signed<int_type>::type;
    using counter = util::counter<signed_int_type>;

    auto l = counter{signed_int_type{-3}};
    auto r = counter{signed_int_type{3}};
    auto range = util::make_range(l, r);
    EXPECT_EQ(range.front(), range.at(0));
    EXPECT_EQ(0, range.at(3));
    EXPECT_EQ(range.back(), range.at(range.size()-1));
    EXPECT_THROW(range.at(range.size()), std::out_of_range);
    EXPECT_THROW(range.at(30), std::out_of_range);
}

TYPED_TEST_P(counter_range, iteration) {
    using int_type = TypeParam;
    using signed_int_type = typename std::make_signed<int_type>::type;
    using counter = util::counter<signed_int_type>;

    auto j = signed_int_type{-3};
    auto l = counter{signed_int_type{j}};
    auto r = counter{signed_int_type{3}};

    for (auto i : util::make_range(l, r)) {
        EXPECT_EQ(j++, i);
    }
}

REGISTER_TYPED_TEST_CASE_P(counter_range, max_size, extreme_size, size, at,
                           iteration);

using int_types = ::testing::Types<signed char, unsigned char, short,
                                   unsigned short, int, unsigned, long,
                                   std::size_t, std::ptrdiff_t>;

using signed_int_types = ::testing::Types<signed char, short,
                                          int, long, std::ptrdiff_t>;

INSTANTIATE_TYPED_TEST_CASE_P(int_types, counter_range, int_types);

TEST(range, assign) {
    const char *cstr = "howdy";
    std::string text = "pardner";

    util::assign(text, util::make_range(cstr, null_terminated));
    EXPECT_EQ("howdy", text);

    const std::vector<char> vstr = {'x', 'y', 'z', 'z', 'y'};
    util::assign(text, vstr);
    EXPECT_EQ("xyzzy", text);

    util::assign_by(text, vstr,
        [](char c) { return c=='z'? '1': '0'; });
    EXPECT_EQ("00110", text);
}

TEST(range, assign_from) {
    int in[] = {0,1,2};

    {
        std::vector<int> copy = util::assign_from(in);
        for (auto i=0u; i<util::size(in); ++i) {
            EXPECT_EQ(in[i], copy[i]);
        }
    }

    {
        std::vector<int> copy = util::assign_from(
            util::transform_view(in, [](int i) {return 2*i;}));
        for (auto i=0u; i<util::size(in); ++i) {
            EXPECT_EQ(2*in[i], copy[i]);
        }
    }
}

TEST(range, sort) {
    char cstr[] = "howdy";

    auto cstr_range = util::make_range(std::begin(cstr), null_terminated);

    // Alas, no forward_iterator sort yet, so make a strict (non-sentinel)
    // range to sort on below

    // simple sort
    util::sort(util::strict_view(cstr_range));
    EXPECT_EQ(std::string("dhowy"), cstr);

    // reverse sort by transform c to -c
    util::sort_by(util::strict_view(cstr_range), [](char c) { return -c; });
    EXPECT_EQ(std::string("ywohd"), cstr);

    // stable sort: move capitals to front, numbers to back
    auto rank = [](char c) {
        return std::isupper(c)? 0: std::isdigit(c)? 2: 1;
    };

    char mixed[] = "t5hH4E3erLL2e1O";
    auto mixed_range = util::make_range(std::begin(mixed), null_terminated);

    util::stable_sort_by(util::strict_view(mixed_range), rank);
    EXPECT_EQ(std::string("HELLOthere54321"), mixed);
}

TEST(range, sum_by) {
    std::string words[] = { "fish", "cakes", "!" };
    auto prepend_ = [](const std::string& x) { return "_"+x; };

    auto result = util::sum_by(words, prepend_);
    EXPECT_EQ("_fish_cakes_!", result);

    result = util::sum_by(words, prepend_, std::string("tasty"));
    EXPECT_EQ("tasty_fish_cakes_!", result);

    auto count = util::sum_by(words, [](const std::string &x) { return x.size(); });
    EXPECT_EQ(10u, count);
}

TEST(range, is_sequence) {
    EXPECT_TRUE(nest::mc::util::is_sequence<std::vector<int>>::value);
    EXPECT_TRUE(nest::mc::util::is_sequence<std::string>::value);
    EXPECT_TRUE(nest::mc::util::is_sequence<int[8]>::value);
}

TEST(range, all_of_any_of) {
    // make a C string into a sentinel-terminated range
    auto cstr = [](const char* s) { return util::make_range(s, null_terminated); };

    // predicate throws on finding 'x' in order to check
    // early stop criterion.
    auto pred = [](char c) { return c=='x'? throw c:c<'5'; };

    // all
    EXPECT_TRUE(util::all_of(std::string(), pred));
    EXPECT_TRUE(util::all_of(std::string("1234"), pred));
    EXPECT_FALSE(util::all_of(std::string("12345"), pred));
    EXPECT_FALSE(util::all_of(std::string("12345x"), pred));

    EXPECT_TRUE(util::all_of(cstr(""), pred));
    EXPECT_TRUE(util::all_of(cstr("1234"), pred));
    EXPECT_FALSE(util::all_of(cstr("12345"), pred));
    EXPECT_FALSE(util::all_of(cstr("12345x"), pred));

    // any
    EXPECT_FALSE(util::any_of(std::string(), pred));
    EXPECT_FALSE(util::any_of(std::string("8765"), pred));
    EXPECT_TRUE(util::any_of(std::string("87654"), pred));
    EXPECT_TRUE(util::any_of(std::string("87654x"), pred));

    EXPECT_FALSE(util::any_of(cstr(""), pred));
    EXPECT_FALSE(util::any_of(cstr("8765"), pred));
    EXPECT_TRUE(util::any_of(cstr("87654"), pred));
    EXPECT_TRUE(util::any_of(cstr("87654x"), pred));
}

#ifdef NMC_HAVE_TBB

TEST(range, tbb_split) {
    constexpr std::size_t N = 20;
    int xs[N];

    for (unsigned i = 0; i<N; ++i) {
        xs[i] = i;
    }

    auto s = util::make_range(&xs[0], &xs[0]+N);

    while (s.size()>1) {
        auto ssize = s.size();
        auto r = decltype(s){s, tbb::split{}};
        EXPECT_GT(r.size(), 0u);
        EXPECT_GT(s.size(), 0u);
        EXPECT_EQ(ssize, r.size()+s.size());
        EXPECT_EQ(s.end(), r.begin());

        EXPECT_TRUE(r.size()>1 || !r.is_divisible());
        EXPECT_TRUE(s.size()>1 || !s.is_divisible());
    }

    for (unsigned i = 1; i<N-1; ++i) {
        s = util::make_range(&xs[0], &xs[0]+N);
        // expect exact splitting by proportion in this instance

        auto r = decltype(s){s, tbb::proportional_split{i, N-i}};
        EXPECT_EQ(&xs[0], s.left);
        EXPECT_EQ(&xs[0]+i, s.right);
        EXPECT_EQ(&xs[0]+i, r.left);
        EXPECT_EQ(&xs[0]+N, r.right);
    }
}

TEST(range, tbb_no_split) {
    std::istringstream sin("10 9 8 7 6");
    auto s = util::make_range(std::istream_iterator<int>(sin), std::istream_iterator<int>());

    EXPECT_FALSE(decltype(s)::is_splittable_in_proportion());
    EXPECT_FALSE(s.is_divisible());
}

#endif
