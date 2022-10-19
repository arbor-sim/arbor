#include <gtest/gtest.h>

#include <string>
#include <arbor/util/pp_util.hpp>

TEST(pp_util, foreach) {
#undef X
#define X(n) #n "."

    std::string str1 = "foo:" ARB_PP_FOREACH(X, a);
    EXPECT_EQ("foo:a.", str1);

#undef LETTERS10
#define LETTERS10 a, b, c, d, e, f, g, h, i, j
    std::string str10 = "bar:" ARB_PP_FOREACH(X, LETTERS10);
    EXPECT_EQ("bar:a.b.c.d.e.f.g.h.i.j.", str10);

    std::string str20 = "baz:" ARB_PP_FOREACH(X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t);
    EXPECT_EQ("baz:a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p.q.r.s.t.", str20);

#undef LETTERS10
#undef X
}
