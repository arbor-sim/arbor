#pragma once

#include <list>
#include <map>
#include <stdexcept>
#include <string>

#include "expression.hpp"

#include "alg_collect.hpp"

using id_prodsum_map = std::map<std::string, alg::prodsum>;

// Given a value expression (e.g. something found on the right hand side
// of an assignment), return the canonical expanded algebraic representation.
// The `exmap` parameter contains the given associations between identifiers and
// algebraic representations.

alg::prodsum expand_expression(Expression* e, const id_prodsum_map& exmap);

// From a sequence of statement expressions, expand all assignments and return
// a map from identifiers to algebraic representations.

template <typename StmtSeq>
id_prodsum_map expand_assignments(const StmtSeq& stmts) {
    using namespace alg;
    id_prodsum_map exmap;

    // This is 'just a test', so don't try to be complete: functions are
    // left unexpanded; procedure calls are ignored.

    for (const auto& stmt: stmts) {
        if (auto assign = stmt->is_assignment()) {
            auto lhs = assign->lhs();
            std::string key;
            if (auto deriv = lhs->is_derivative()) {
                key = deriv->spelling()+"'";
            }
            else if (auto id = lhs->is_identifier()) {
                key = id->spelling();
            }
            else {
                // don't know what we have here! skip.
                continue;
            }

            exmap[key] = expand_expression(assign->rhs(), exmap);
        }
    }
    return exmap;
}
