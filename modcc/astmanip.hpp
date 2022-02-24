#pragma once

// Helper utilities for manipulating/modifying AST.

#include <string>

#include "expression.hpp"
#include "location.hpp"
#include "scope.hpp"
#include <libmodcc/export.hpp>

// Create new local variable symbol and local declaration expression in current scope.
// Returns the local declaration expression.

struct local_declaration {
    expression_ptr local_decl;
    expression_ptr id;
    scope_ptr scope;
};

ARB_LIBMODCC_API local_declaration make_unique_local_decl(
    scope_ptr scope,
    Location loc,
    std::string const& prefix="ll");

// Create a local declaration as for `make_unique_local_decl`, together with an
// assignment to it from the given expression, using the location of that expression.
// Returns local declaration expression, assignment expression, new identifier id and
// consequent scope.

struct local_assignment {
    expression_ptr local_decl;
    expression_ptr assignment;
    expression_ptr id;
    scope_ptr scope;
};

ARB_LIBMODCC_API local_assignment make_unique_local_assign(
    scope_ptr scope,
    Expression* e,
    std::string const& prefix="ll");

inline local_assignment make_unique_local_assign(
    scope_ptr scope,
    expression_ptr& e,
    std::string const& prefix="ll")
{
    return make_unique_local_assign(scope, e.get(), prefix);
}

