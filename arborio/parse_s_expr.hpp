#pragma once

#include <arbor/s_expr.hpp>
#include <arborio/export.hpp>
#include <arborio/cableio.hpp>

namespace arborio {
using namespace arb;

// Helper function for programmatically building lists
//     slist(1, 2, "hello world", "banjax@cat/3"_symbol);
// Produces the following s-expression:
//     (1 2 "hello world" banjax@cat/3)
// Can be nested:
//     slist(1, slist(2, 3), 4, 5 );
// Produces:
//     (1 (2 3) 4 5)

template <typename T>
s_expr slist(T v) {
    return {v, {}};
}

template <typename T, typename... Args>
s_expr slist(T v, Args... args) {
    return {v, slist(args...)};
}

inline s_expr slist() {
    return {};
}

template <typename I, typename S>
s_expr slist_range(I b, S e) {
    return b==e ? s_expr{}
                : s_expr{*b, slist_range(++b,e)};
}

template <typename Range>
s_expr slist_range(const Range& range) {
    return slist_range(std::begin(range), std::end(range));
}

ARB_ARBORIO_API parse_hopefully<std::any> parse_expression(const std::string&);

} // namespace arborio
