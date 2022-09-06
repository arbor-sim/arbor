#pragma once

// Perform naive symbolic differenation on a (rhs) expression;
// treat all identifiers as independent, and function calls
// with the variable in argument as opaque.
//
// This is just for linearity and possibly polynomiality testing, so
// don't try too hard.

#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "expression.hpp"
#include <libmodcc/export.hpp>


// True if `id` matches the spelling of any identifier in the expression.
ARB_LIBMODCC_API bool involves_identifier(Expression* e, const std::string& id);

using identifier_set = std::vector<std::string>;
ARB_LIBMODCC_API bool involves_identifier(Expression* e, const identifier_set& ids);

// Return new expression formed by folding constants and removing trivial terms.
ARB_LIBMODCC_API expression_ptr constant_simplify(Expression* e);

// Extract value of expression that is a NumberExpression, or else return NAN.
ARB_LIBMODCC_API double expr_value(Expression* e);

// Test if expression is a NumberExpression with value zero.
inline bool is_zero(Expression* e) {
    return expr_value(e)==0;
}

// Return new expression of symbolic partial differentiation of argument wrt `id`.
ARB_LIBMODCC_API expression_ptr symbolic_pdiff(Expression* e, const std::string& id);

// Substitute all occurances of identifier `id` within expression by a clone of `sub`.
// (Only applicable to unary, binary, call and number expressions.)
ARB_LIBMODCC_API expression_ptr substitute(Expression* e, const std::string& id, Expression* sub);

using substitute_map = std::map<std::string, expression_ptr>;
ARB_LIBMODCC_API expression_ptr substitute(Expression* e, const substitute_map& sub);

// Convenience interfaces for the above functions work with `expression_ptr` as
// well as with `Expression*` values.

inline bool involves_identifier(const expression_ptr& e, const std::string& id) {
    return involves_identifier(e.get(), id);
}

inline bool involves_identifier(const expression_ptr& e, const identifier_set& ids) {
    return involves_identifier(e.get(), ids);
}

inline expression_ptr constant_simplify(const expression_ptr& e) {
    return constant_simplify(e.get());
}

inline double expr_value(const expression_ptr& e) {
    return expr_value(e.get());
}

inline double is_zero(const expression_ptr& e) {
    return is_zero(e.get());
}

inline expression_ptr symbolic_pdiff(const expression_ptr& e, const std::string& id) {
    return symbolic_pdiff(e.get(), id);
}

inline expression_ptr substitute(const expression_ptr& e, const std::string& id, const expression_ptr& sub) {
    return substitute(e.get(), id, sub.get());
}

inline expression_ptr substitute(const expression_ptr& e, const substitute_map& sub) {
    return substitute(e.get(), sub);
}


// Linearity testing

struct linear_test_result: public error_stack {
    bool is_linear = false;
    bool is_homogeneous = false;
    bool is_constant = true;
    expression_ptr constant;
    std::map<std::string, expression_ptr> coef;

    bool monolinear() const {
        unsigned nlinear = 0;
        for (auto& entry: coef) {
            if (!is_zero(entry.second) && ++nlinear>1) return false;
        }
        return true;
    }

    bool monolinear(const std::string& var) const {
        for (auto& entry: coef) {
            if (!is_zero(entry.second) && var!=entry.first) return false;
        }
        return true;
    }

    friend std::ostream& operator<<(std::ostream& o, const linear_test_result& r) {
        o << "{linear: " << r.is_linear << "; homogeneous: " << r.is_homogeneous << "\n";
        o << " constant term: " << r.constant->to_string();
        for (const auto& p: r.coef) {
            o << "\n coef " << p.first << ": " << p.second->to_string();
        }
        o << "}";
        return o;
    }
};

ARB_LIBMODCC_API linear_test_result linear_test(Expression* e, const std::vector<std::string>& vars);

inline linear_test_result linear_test(const expression_ptr& e, const std::vector<std::string>& vars) {
    return linear_test(e.get(), vars);
}
