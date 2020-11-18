#pragma once

/*
 * Macro definitions for defining comparison operators for
 * record-like types.
 *
 * Use:
 *
 * To define comparison operations for a record type xyzzy
 * with fields foo, bar and baz:
 *
 * ARB_DEFINE_LEXICOGRAPHIC_ORDERING(xyzzy,(a.foo,a.bar,a.baz),(b.foo,b.bar,b.baz))
 *
 * The explicit use of 'a' and 'b' in the second and third parameters
 * is needed only to save a heroic amount of preprocessor macro
 * deep magic.
 *
 */

#include <tuple>

#define ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(proxy,op,type,a_fields,b_fields) \
inline bool operator op(const type& a,const type& b) { return proxy a_fields op proxy b_fields; }

#define ARB_DEFINE_LEXICOGRAPHIC_ORDERING(type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::tie,<,type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::tie,>,type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::tie,<=,type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::tie,>=,type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::tie,!=,type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::tie,==,type,a_fields,b_fields)

#define ARB_DEFINE_LEXICOGRAPHIC_ORDERING_BY_VALUE(type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::make_tuple,<,type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::make_tuple,>,type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::make_tuple,<=,type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::make_tuple,>=,type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::make_tuple,!=,type,a_fields,b_fields) \
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_IMPL_(std::make_tuple,==,type,a_fields,b_fields)

