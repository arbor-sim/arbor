#include <iostream>
#include <cmath>

#include "error.hpp"
#include "expressionclassifier.hpp"
#include "util.hpp"

// this turns out to be quite easy, however quite fiddly to do right.

// default is to do nothing and return
void ExpressionClassifierVisitor::visit(Expression *e) {
    throw compiler_exception(" attempting to apply linear analysis on " + e->to_string(), e->location());
}

// number expresssion
void ExpressionClassifierVisitor::visit(NumberExpression *e) {
    // save the coefficient as the number
    coefficient_ = e->clone();
}

// identifier expresssion
void ExpressionClassifierVisitor::visit(IdentifierExpression *e) {
    // check if symbol of identifier matches the identifier
    if(symbol_ == e->symbol()) {
        found_symbol_ = true;
        coefficient_.reset(new NumberExpression(Location(), "1"));
    }
    else {
        coefficient_ = e->clone();
    }
}

/// unary expresssion
void ExpressionClassifierVisitor::visit(UnaryExpression *e) {
    e->expression()->accept(this);
    if(found_symbol_) {
        switch(e->op()) {
            // plus or minus don't change linearity
            case tok::minus :
                coefficient_ = unary_expression(Location(),
                                                e->op(),
                                                std::move(coefficient_));
                return;
            case tok::plus :
                return;
            // one of these applied to the symbol certainly isn't linear
            case tok::exp :
            case tok::cos :
            case tok::sin :
            case tok::log :
                is_linear_ = false;
                return;
            default :
                throw compiler_exception(
                    "attempting to apply linear analysis on unsuported UnaryExpression "
                    + yellow(token_string(e->op())), e->location());
        }
    }
    else {
        coefficient_ = e->clone();
    }
}

// binary expresssion
// handle all binary expressions with one routine, because the
// pre-order and in-order code is the same for all cases
void ExpressionClassifierVisitor::visit(BinaryExpression *e) {
    bool lhs_contains_symbol = false;
    bool rhs_contains_symbol = false;
    expression_ptr lhs_coefficient;
    expression_ptr rhs_coefficient;
    expression_ptr lhs_constant;
    expression_ptr rhs_constant;

    // check the lhs
    reset();
    e->lhs()->accept(this);
    lhs_contains_symbol = found_symbol_;
    lhs_coefficient     = std::move(coefficient_);
    lhs_constant        = std::move(constant_);
    if(!is_linear_) return; // early return if nonlinear

    // check the rhs
    reset();
    e->rhs()->accept(this);
    rhs_contains_symbol = found_symbol_;
    rhs_coefficient     = std::move(coefficient_);
    rhs_constant        = std::move(constant_);
    if(!is_linear_) return; // early return if nonlinear

    // mark symbol as found if in either lhs or rhs
    found_symbol_ = rhs_contains_symbol || lhs_contains_symbol;

    if( found_symbol_ ) {
        // if both lhs and rhs contain symbol check that the binary operator
        // preserves linearity
        // note that we don't have to test for linearity, because we abort early
        // if either lhs or rhs are nonlinear
        if( rhs_contains_symbol && lhs_contains_symbol ) {
            // be careful to get the order of operation right for
            // non-computative operators
            switch(e->op()) {
                // addition and subtraction are valid, nothing else is
                case tok::plus :
                case tok::minus :
                    coefficient_ =
                        binary_expression(Location(),
                                          e->op(),
                                          std::move(lhs_coefficient),
                                          std::move(rhs_coefficient));
                    return;
                // multiplying two expressions that depend on symbol is nonlinear
                case tok::times :
                case tok::pow   :
                case tok::divide :
                default         :
                    is_linear_ = false;
                    return;
            }
        }
        // special cases :
        //      operator    | invalid symbol location
        //      -------------------------------------
        //      pow         | lhs OR rhs
        //      comparisons | lhs OR rhs
        //      division    | rhs
        ////////////////////////////////////////////////////////////////////////
        // only RHS contains the symbol
        ////////////////////////////////////////////////////////////////////////
        else if(rhs_contains_symbol) {
            switch(e->op()) {
                case tok::times  :
                    // determine the linear coefficient
                    if( rhs_coefficient->is_number() &&
                        rhs_coefficient->is_number()->value()==1) {
                        coefficient_ = lhs_coefficient->clone();
                    }
                    else {
                        coefficient_ =
                            binary_expression(Location(),
                                              tok::times,
                                              lhs_coefficient->clone(),
                                              rhs_coefficient->clone());
                    }
                    // determine the constant
                    if(rhs_constant) {
                        constant_ =
                            binary_expression(Location(),
                                              tok::times,
                                              std::move(lhs_coefficient),
                                              std::move(rhs_constant));
                    } else {
                        constant_ = nullptr;
                    }
                    return;
                case tok::plus :
                    // constant term
                    if(lhs_constant && rhs_constant) {
                        constant_ =
                            binary_expression(Location(),
                                              tok::plus,
                                              std::move(lhs_constant),
                                              std::move(rhs_constant));
                    }
                    else if(rhs_constant) {
                        constant_ = binary_expression(Location(),
                                                      tok::plus,
                                                      std::move(lhs_coefficient),
                                                      std::move(rhs_constant));
                    }
                    else {
                        constant_ = std::move(lhs_coefficient);
                    }
                    // coefficient
                    coefficient_ = std::move(rhs_coefficient);
                    return;
                case tok::minus :
                    // constant term
                    if(lhs_constant && rhs_constant) {
                        constant_ = binary_expression(Location(),
                                                      tok::minus,
                                                      std::move(lhs_constant),
                                                      std::move(rhs_constant));
                    }
                    else if(rhs_constant) {
                        constant_ = binary_expression(Location(),
                                                      tok::minus,
                                                      std::move(lhs_coefficient),
                                                      std::move(rhs_constant));
                    }
                    else {
                        constant_ = std::move(lhs_coefficient);
                    }
                    // coefficient
                    coefficient_ = unary_expression(Location(),
                                                    e->op(),
                                                    std::move(rhs_coefficient));
                    return;
                case tok::pow    :
                case tok::divide :
                case tok::lt     :
                case tok::lte    :
                case tok::gt     :
                case tok::gte    :
                case tok::equality :
                    is_linear_ = false;
                    return;
                default:
                    return;
            }
        }
        ////////////////////////////////////////////////////////////////////////
        // only LHS contains the symbol
        ////////////////////////////////////////////////////////////////////////
        else if(lhs_contains_symbol) {
            switch(e->op()) {
                case tok::times  :
                    // check if the lhs is == 1
                    if( lhs_coefficient->is_number() &&
                        lhs_coefficient->is_number()->value()==1) {
                        coefficient_ = rhs_coefficient->clone();
                    }
                    else {
                        coefficient_ =
                            binary_expression(Location(),
                                              tok::times,
                                              std::move(lhs_coefficient),
                                              std::move(rhs_coefficient));
                    }
                    // constant term
                    if(lhs_constant) {
                        constant_ = binary_expression(Location(),
                                                      tok::times,
                                                      std::move(lhs_constant),
                                                      std::move(rhs_coefficient));
                    } else {
                        constant_ = nullptr;
                    }
                    return;
                case tok::plus  :
                    coefficient_ = std::move(lhs_coefficient);
                    // constant term
                    if(lhs_constant && rhs_constant) {
                        constant_ = binary_expression(Location(),
                                                      tok::plus,
                                                      std::move(lhs_constant),
                                                      std::move(rhs_constant));
                    }
                    else if(lhs_constant) {
                        constant_ = binary_expression(Location(),
                                                      tok::plus,
                                                      std::move(lhs_constant),
                                                      std::move(rhs_coefficient));
                    }
                    else {
                        constant_ = std::move(rhs_coefficient);
                    }
                    return;
                case tok::minus :
                    coefficient_ = std::move(lhs_coefficient);
                    // constant term
                    if(lhs_constant && rhs_constant) {
                        constant_ = binary_expression(Location(),
                                                      tok::minus,
                                                      std::move(lhs_constant),
                                                      std::move(rhs_constant));
                    }
                    else if(lhs_constant) {
                        constant_ = binary_expression(Location(),
                                                      tok::minus,
                                                      std::move(lhs_constant),
                                                      std::move(rhs_coefficient));
                    }
                    else {
                        constant_ = unary_expression(Location(),
                                                     tok::minus,
                                                     std::move(rhs_coefficient));
                    }
                    return;
                case tok::divide:
                    coefficient_ = binary_expression(Location(),
                                                     tok::divide,
                                                     std::move(lhs_coefficient),
                                                     rhs_coefficient->clone());
                    if(lhs_constant) {
                        constant_ = binary_expression(Location(),
                                                      tok::divide,
                                                      std::move(lhs_constant),
                                                      std::move(rhs_coefficient));
                    }
                    return;
                case tok::pow    :
                case tok::lt     :
                case tok::lte    :
                case tok::gt     :
                case tok::gte    :
                case tok::equality :
                    is_linear_ = false;
                    return;
                default:
                    return;
            }
        }
    }
    // neither lhs or rhs contains symbol
    // continue building the coefficient
    else {
        coefficient_ = e->clone();
    }
}

void ExpressionClassifierVisitor::visit(CallExpression *e) {
    for(auto& a : e->args()) {
        a->accept(this);
        // we assume that the parameter passed into a function
        // won't be linear
        if(found_symbol_) {
            is_linear_ = false;
            return;
        }
    }
}

