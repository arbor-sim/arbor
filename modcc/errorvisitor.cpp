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
    e->condition()->accept(this);
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

// call expresssion
void ErrorVisitor::visit(CallExpression *e) {
    for(auto& expression: e->args()) {
        expression->accept(this);
    }
    print_error(e);
}

// reaction expresssion
void ErrorVisitor::visit(ReactionExpression *e) {
    e->lhs()->accept(this);
    e->rhs()->accept(this);
    e->fwd_rate()->accept(this);
    e->rev_rate()->accept(this);
    print_error(e);
}

// stoich expresssion
void ErrorVisitor::visit(StoichExpression *e) {
    for (auto& expression: e->terms()) {
        expression->accept(this);
    }
    print_error(e);
}

// stoich term expresssion
void ErrorVisitor::visit(StoichTermExpression *e) {
    e->ident()->accept(this);
    e->coeff()->accept(this);
    print_error(e);
}

// compartment expresssion
void ErrorVisitor::visit(CompartmentExpression *e) {
    e->scale_factor()->accept(this);
    for (auto& expression: e->state_vars()) {
        expression->accept(this);
    }
    print_error(e);
}

// pdiff expresssion
void ErrorVisitor::visit(PDiffExpression *e) {
    e->var()->accept(this);
    e->arg()->accept(this);
    print_error(e);
}