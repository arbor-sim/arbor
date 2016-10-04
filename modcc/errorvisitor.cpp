#include "errorvisitor.hpp"

/*
 * we use a post order walk to print the erros in an expression after those
 * in all of its children
 */

void ErrorVisitor::visit(Expression *e) {
    print_error(e);
}

// traverse the statements in a procedure
void ErrorVisitor::visit(ProcedureExpression *e) {
    for(auto& expression : e->args()) {
        expression->accept(this);
    }

    e->body()->accept(this);
    print_error(e);
}

// traverse the statements in a function
void ErrorVisitor::visit(FunctionExpression *e) {
    for(auto& expression : e->args()) {
        expression->accept(this);
    }

    e->body()->accept(this);
    print_error(e);
}

// an if statement
void ErrorVisitor::visit(IfExpression *e) {
    e->true_branch()->accept(this);
    if(e->false_branch()) {
        e->false_branch()->accept(this);
    }

    print_error(e);
}

void ErrorVisitor::visit(BlockExpression* e) {
    for(auto& expression : e->statements()) {
        expression->accept(this);
    }

    print_error(e);
}

void ErrorVisitor::visit(InitialBlock* e) {
    for(auto& expression : e->statements()) {
        expression->accept(this);
    }

    print_error(e);
}

// unary expresssion
void ErrorVisitor::visit(UnaryExpression *e) {
    e->expression()->accept(this);
    print_error(e);
}

// binary expresssion
void ErrorVisitor::visit(BinaryExpression *e) {
    e->lhs()->accept(this);
    e->rhs()->accept(this);
    print_error(e);
}

// binary expresssion
void ErrorVisitor::visit(CallExpression *e) {
    for(auto& expression: e->args()) {
        expression->accept(this);
    }
    print_error(e);
}

