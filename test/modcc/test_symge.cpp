#include <map>
#include <utility>
#include <vector>

#include "symge.hpp"
#include "common.hpp"

using namespace symge;

TEST(symge, table_define) {
    symbol_table tbl;

    EXPECT_EQ(0u, tbl.size());

    auto s1 = tbl.define();
    auto s2 = tbl.define("foo");
    auto s3 = tbl.define(symbol_term_diff{});
    auto s4 = tbl.define("bar",symbol_term_diff{});

    EXPECT_EQ(4u, tbl.size());
    EXPECT_TRUE(tbl.valid(s1));
    EXPECT_TRUE(tbl.valid(s2));
    EXPECT_TRUE(tbl.valid(s3));
    EXPECT_TRUE(tbl.valid(s4));
    EXPECT_FALSE(tbl.valid(symbol{}));
}

TEST(symge, symbol_ops) {
    symbol_table tbl;

    symbol a = tbl.define("a");
    symbol b = tbl.define("b");
    symbol c = tbl.define("c", a*b);
    symbol d = tbl.define("d", -(a*b));
    symbol e = tbl.define("e", c*d-a*b);

    EXPECT_TRUE(tbl.primitive(a));
    EXPECT_TRUE(tbl.primitive(b));
    EXPECT_FALSE(tbl.primitive(c));
    EXPECT_FALSE(tbl.primitive(d));
    EXPECT_FALSE(tbl.primitive(e));

    auto cdef = tbl.get(c);
    EXPECT_FALSE(cdef.left.is_zero());
    EXPECT_TRUE(cdef.right.is_zero());

    auto ddef = tbl.get(d);
    EXPECT_TRUE(ddef.left.is_zero());
    EXPECT_FALSE(ddef.right.is_zero());

    auto edef = tbl.get(e);
    EXPECT_FALSE(edef.left.is_zero());
    EXPECT_FALSE(edef.right.is_zero());

    EXPECT_EQ(c, edef.left.left);
    EXPECT_EQ(d, edef.left.right);
    EXPECT_EQ(a, edef.right.left);
    EXPECT_EQ(b, edef.right.right);
}

TEST(symge, symbol_name) {
    symbol_table tbl;

    symbol a = tbl.define("a");
    EXPECT_EQ("a", tbl.name(a));
    EXPECT_EQ("a", name(a));
    tbl.name(a, "b");
    EXPECT_EQ("b", name(a));
}

TEST(symge, table_index) {
    symbol_table tbl;

    symbol a = tbl.define("a");
    symbol b = tbl.define("b");
    symbol c = tbl.define("c", a*b);
    symbol d = tbl.define("d", -(a*b));
    symbol e = tbl.define("e", c*d-a*b);

    EXPECT_EQ(a, tbl[0]);
    EXPECT_EQ(d, tbl[3]);
    EXPECT_EQ(e, tbl[4]);
}

#include <iostream>

struct value_store {
    mutable std::map<std::string, double> values;

    std::string new_var() {
        return "v"+std::to_string(values.size());
    }

    double& assign(symbol s, double v) {
        if (name(s).empty()) {
            const_cast<symbol_table*>(s.table())->name(s, new_var());
        }

        return values[name(s)] = v;
    }

    double eval(symbol s) {
        return values[name(s)];
    }

    double eval(symbol_term t) {
        return t.is_zero()? 0.0: eval(t.left)*eval(t.right);
    }

    double eval(symbol_term_diff d) {
        return eval(d.left)-eval(d.right);
    }

    double& operator[](symbol s) {
        if (name(s).empty() || !values.count(name(s))) {
            return assign(s, 0.0);
        }
        return values[name(s)];
    }
};


TEST(symge, gj_reduce_3x3) {
    // solve:
    //
    // | 2  0  3 | | x |   | 6 |
    // | 0  4  0 | | y | = | 7 |
    // | 0 -1  5 | | z |   | 8 |
    //
    // with expected answer:
    //
    // x = 3/40; y = 7/4; z = 39/20

    symbol_table tbl;
    auto a = tbl.define("a");
    auto b = tbl.define("b");
    auto c = tbl.define("c");
    auto d = tbl.define("d");
    auto e = tbl.define("e");
    auto p = tbl.define("p");
    auto q = tbl.define("q");
    auto r = tbl.define("r");

    sym_matrix A(3,3);
    A[0] = sym_row({{0, a}, {2, b}});
    A[1] = sym_row({{1, c}});
    A[2] = sym_row({{1, d}, {2, e}});

    std::vector<symbol> B = { p, q, r };
    A.augment(B);

    gj_reduce(A, tbl);

    value_store v;
    v[a] = 2;
    v[b] = 3;
    v[c] = 4;
    v[d] = -1;
    v[e] = 5;
    v[p] = 6;
    v[q] = 7;
    v[r] = 8;

    for (unsigned i = 0; i<tbl.size(); ++i) {
        symbol s = tbl[i];
        if (!primitive(s)) {
            v.assign(s, v.eval(definition(s)));
        }
    }

    double x = v.eval(A[0][3])/v.eval(A[0][0]);
    double y = v.eval(A[1][3])/v.eval(A[1][1]);
    double z = v.eval(A[2][3])/v.eval(A[2][2]);

    EXPECT_NEAR(x, 3.0/40.0, 1e-6);
    EXPECT_NEAR(y, 7.0/4.0, 1e-6);
    EXPECT_NEAR(z, 39.0/20.0, 1e-6);
}
