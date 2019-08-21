#include <utility>

#include "common.hpp"
#include "msparse.hpp"

using drow = msparse::row<double>;
using dmatrix = msparse::matrix<double>;
using msparse::row_npos;

namespace msparse {
bool operator==(const drow::entry& a, const drow::entry& b) {
    return a.col==b.col && a.value==b.value;
}
}

TEST(msparse, row_ctor) {

    drow r1;
    EXPECT_TRUE(r1.empty());

    drow r2({{0,3.0},{2,-1.5}});
    EXPECT_FALSE(r2.empty());
    EXPECT_EQ(2u, r2.size());

    drow r3(r2);
    EXPECT_FALSE(r3.empty());
    EXPECT_EQ(2u, r3.size());

    drow::entry entries[] = { {1,2.}, {4,-2.}, {5,0} };
    drow r4(std::begin(entries), std::end(entries));
    EXPECT_FALSE(r4.empty());
    EXPECT_EQ(3u, r4.size());

    ASSERT_THROW(drow({{2,-1.5}, {0,3.0}}), msparse::msparse_error);

    drow r5 = r4;
    EXPECT_FALSE(r5.empty());
    EXPECT_EQ(3u, r5.size());
}

TEST(msparse, row_iter) {
    using drow = msparse::row<double>;

    drow r1;
    EXPECT_EQ(r1.begin(), r1.end());
    const drow& r1c = r1;
    EXPECT_EQ(r1c.begin(), r1c.end());

    drow r2({{0,3.0},{2,-1.5}});
    EXPECT_EQ(*r2.begin(), drow::entry({0,3.0}));
    EXPECT_EQ(*std::prev(r2.end()), drow::entry({2,-1.5}));
    EXPECT_EQ(2u, std::distance(r2.begin(), r2.end()));
}

TEST(msparse, row_query) {
    EXPECT_EQ(row_npos, drow{}.mincol());
    EXPECT_EQ(row_npos, drow{}.maxcol());
    EXPECT_EQ(row_npos, drow{}.index(2u));

    drow r({{1,2.0}, {2,4.0}, {4,7.0}, {6,9.0}});

    EXPECT_EQ(4u, r.size());
    EXPECT_EQ(1u, r.mincol());
    EXPECT_EQ(6u, r.maxcol());

    EXPECT_EQ(2u, r.mincol_after(1));
    EXPECT_EQ(4u, r.mincol_after(2));
    EXPECT_EQ(4u, r.mincol_after(3));
    EXPECT_EQ(row_npos, r.mincol_after(6));

    EXPECT_EQ(0u, r.index(1));
    EXPECT_EQ(1u, r.index(2));
    EXPECT_EQ(3u, r.index(6));
    EXPECT_EQ(row_npos, r.index(0));
    EXPECT_EQ(row_npos, r.index(5));
    EXPECT_EQ(row_npos, r.index(7));

    EXPECT_EQ(0.0, r[0]);
    EXPECT_EQ(2.0, r[1]);
    EXPECT_EQ(4.0, r[2]);
    EXPECT_EQ(0.0, r[3]);
    EXPECT_EQ(9.0, r[6]);
    EXPECT_EQ(0.0, r[7]);

    EXPECT_EQ(drow::entry({1,2.0}), r.get(0));
    EXPECT_EQ(drow::entry({2,4.0}), r.get(1));
    EXPECT_EQ(drow::entry({4,7.0}), r.get(2));
}

TEST(msparse, row_mutate) {
    drow r;

    r.push_back({1, 2.0});
    EXPECT_EQ(1u, r.size());

    r.push_back({3, 3.0});
    EXPECT_EQ(2u, r.size());

    ASSERT_THROW(r.push_back({3, 2.0}), msparse::msparse_error);
    ASSERT_THROW(r.push_back({2, 2.0}), msparse::msparse_error);

    r.truncate(4);
    EXPECT_EQ(2u, r.size());

    r.truncate(3);
    EXPECT_EQ(1u, r.size());

    r.truncate(0);
    EXPECT_EQ(0u, r.size());

    r[7] = 4.0;
    EXPECT_EQ(1u, r.size());
    EXPECT_EQ(7u, r.mincol());
}

TEST(msparse, matrix_ctor) {
    dmatrix M;
    EXPECT_EQ(0u, M.size());
    EXPECT_EQ(0u, M.nrow());
    EXPECT_EQ(0u, M.ncol());
    EXPECT_TRUE(M.empty());

    dmatrix M2(5,3);
    EXPECT_EQ(5u, M2.size());
    EXPECT_EQ(5u, M2.nrow());
    EXPECT_EQ(3u, M2.ncol());
    EXPECT_FALSE(M2.empty());

    dmatrix M3(M2);
    EXPECT_EQ(5u, M3.nrow());
    EXPECT_EQ(3u, M3.ncol());

    M2.clear();
    EXPECT_EQ(0u, M2.size());
    EXPECT_EQ(0u, M2.nrow());
    EXPECT_EQ(0u, M2.ncol());
    EXPECT_TRUE(M2.empty());
}

TEST(msparse, matrix_index) {
    dmatrix M2(5,3);
    M2[4].push_back({2, -1.0});

    const dmatrix& M2c = M2;

    EXPECT_EQ(-1.0, M2[4][2]);
    EXPECT_EQ(-1.0, M2c[4][2]);
}

TEST(msparse, matrix_augment) {
    dmatrix M(5,3);

    M[1][2] = 9.0;
    EXPECT_FALSE(M.augmented());
    EXPECT_EQ(row_npos, M.augcol());

    EXPECT_EQ(0u, M[0].size());
    EXPECT_EQ(1u, M[1].size());

    double aug1[] = {5, 4, 3, 2, 1};
    M.augment(aug1);

    // short is okay...
    double aug2[] = {1, 2, 1};
    M.augment(aug2);

    EXPECT_TRUE(M.augmented());
    EXPECT_EQ(3u, M.augcol());

    EXPECT_EQ(2u, M[0].size());
    EXPECT_EQ(3u, M[0].mincol());
    EXPECT_EQ(4u, M[0].maxcol());

    EXPECT_EQ(3u, M[1].size());
    EXPECT_EQ(2u, M[1].mincol());
    EXPECT_EQ(4u, M[1].maxcol());

    EXPECT_EQ(9.0, M[1][2]);
    EXPECT_EQ(4.0, M[1][3]);
    EXPECT_EQ(2.0, M[1][4]);

    M.diminish();

    EXPECT_FALSE(M.augmented());
    EXPECT_EQ(row_npos, M.augcol());

    EXPECT_EQ(0u, M[0].size());
    EXPECT_EQ(1u, M[1].size());
    EXPECT_EQ(9.0, M[1][M[1].maxcol()]);

    // augmenting with too long a column is not okay...
    double aug3[] = {1, 2, 3, 4, 5, 6};
    ASSERT_THROW(M.augment(aug3), msparse::msparse_error);
}
