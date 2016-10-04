#include <iostream>
#include <cmath>

#include "constantfolder.hpp"

/*
   perform a walk of the AST
   - pre-order : mark node as not a number
   - in-order  : convert all children that marked themselves as numbers into NumberExpressions
   - post-order: mark the current node as a constant if all of its children
                 were converted to NumberExpressions

   all calculations and intermediate results use 80 bit floating point precision (long double)
*/

// default is to do nothing and return
void ConstantFolderVisitor::visit(Expression *e) {
    is_number = false;
}

// number expresssion
void ConstantFolderVisitor::visit(NumberExpression *e) {
    // set constant number and return
    is_number = true;
    value = e->value();
}

/// unary expresssion
void ConstantFolderVisitor::visit(UnaryExpression *e) {
    is_number = false;
    e->expression()->accept(this);
    if(is_number) {
        if(!e->is_number()) {
            e->replace_expression(make_expression<NumberExpression>(e->location(), value));
        }
        switch(e->op()) {
            case tok::minus :
                value = -value;
                return;
            case tok::exp :
                value = std::exp(value);
                return;
            case tok::cos :
                value = std::cos(value);
                return;
            case tok::sin :
                value = std::sin(value);
                return;
            case tok::log :
                value = std::log(value);
                return;
            default :
                throw compiler_exception(
                    "attempting constant folding on unsuported unary operator "
                    + yellow(token_string(e->op())),
                    e->location());
        }
    }
}

// binary expresssion
// handle all binary expressions with one routine, because the
// pre-order and in-order code is the same for all cases
void ConstantFolderVisitor::visit(BinaryExpression *e) {
    bool lhs_is_number = false;
    long double lhs_value  = 0;

    // check the lhs
    is_number = false;
    e->lhs()->accept(this);
    if(is_number) {
        lhs_value = value;
        lhs_is_number = true;
        // replace lhs with a number node, if it is not already one
        if(!e->lhs()->is_number()) {
            e->replace_lhs( make_expression<NumberExpression>(e->location(), value) );
        }
    }
    //std::cout << "lhs : " << e->lhs()->to_string() << std::endl;

    // check the rhs
    is_number = false;
    e->rhs()->accept(this);
    if(is_number) {
        // replace rhs with a number node, if it is not already one
        if(!e->rhs()->is_number()) {
            //std::cout << "rhs : " << e->rhs()->to_string() << " -> ";
            e->replace_rhs( make_expression<NumberExpression>(e->location(), value) );
            //std::cout << e->rhs()->to_string() << std::endl;
        }
    }
    //std::cout << "rhs : " << e->rhs()->to_string() << std::endl;

    auto rhs_is_number = is_number;
    is_number = rhs_is_number && lhs_is_number;

    // check to see if both lhs and rhs are numbers
    // mark this node as a number if so
    if(is_number) {
        // be careful to get the order of operation right for
        // non-computative operators
        switch(e->op()) {
            case tok::plus :
                value = lhs_value + value;
                return;
            case tok::minus :
                value = lhs_value - value;
                return;
            case tok::times :
                value = lhs_value * value;
                return;
            case tok::divide :
                value = lhs_value / value;
                return;
            case tok::pow :
                value = std::pow(lhs_value, value);
                return;
            // don't fold comparison operators (we have no internal support
            // for boolean values in nodes). leave for the back end compiler.
            // not a big deal, because these are not counted when estimating
            // flops with the FLOP visitor
            case tok::lt     :
            case tok::lte    :
            case tok::gt     :
            case tok::gte    :
            case tok::equality :
                is_number = false;
                return;
            default         :
                throw compiler_exception(
                    "attempting constant folding on unsuported binary operator "
                    + yellow(token_string(e->op())),
                    e->location());
        }
    }
}

void ConstantFolderVisitor::visit(CallExpression *e) {
    is_number = false;
    for(auto& a : e->args()) {
        a->accept(this);
        if(is_number) {
            // replace rhs with a number node, if it is not already one
            if(!a->is_number()) {
                a.reset(new NumberExpression(a->location(), value));
            }
        }
    }
}

void ConstantFolderVisitor::visit(BlockExpression *e) {
    is_number = false;
    for(auto &expression : e->statements()) {
        expression->accept(this);
    }
}

void ConstantFolderVisitor::visit(FunctionExpression *e) {
    is_number = false;
    e->body()->accept(this);
}

void ConstantFolderVisitor::visit(ProcedureExpression *e) {
    is_number = false;
    e->body()->accept(this);
}

void ConstantFolderVisitor::visit(IfExpression *e) {
    is_number = false;
    e->condition()->accept(this);
    e->true_branch()->accept(this);
    if(e->false_branch()) {
        e->false_branch()->accept(this);
    }
}

