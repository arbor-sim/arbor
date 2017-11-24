#include <ostream>
#include <unordered_map>

#include "cexpr_render.hpp"
#include "error.hpp"

void CExprRenderer::render_as_call(const char* sub, Expression* e) {
    out_ << sub << '(';
    e->accept(this);
    out_ << ')';
}

void CExprRenderer::render_as_call(const char* sub, Expression* e1, Expression* e2) {
    out_ << sub << '(';
    e1->accept(this);
    out_ << ", ";
    e2->accept(this);
    out_ << ')';
}

void CExprRenderer::visit(NumberExpression* e) {
    out_ << " " << e->value();
}

void CExprRenderer::visit(UnaryExpression* e) {
    // Place a space in front of minus sign to avoid invalid
    // expressions of the form: (v[i]--67)
    static std::unordered_map<tok, const char*> unaryop_tbl = {
        {tok::minus, " -"},
        {tok::exp,   "exp"},
        {tok::cos,   "cos"},
        {tok::sin,   "sin"},
        {tok::log,   "log"}
    };

    if (!unaryop_tbl.count(e->op())) {
        throw compiler_exception(
            "CExprRender: unsupported unary operator "+token_string(e->op()), e->location());
    }

    const char* op_spelling = unaryop_tbl.at(e->op());
    Expression* inner = e->expression();

    // No need to use parenthesis for unary minus if inner expression is
    // not binary.
    if (e->op()==tok::minus && !inner->is_binary()) {
        out_ << op_spelling;
        inner->accept(this);
    }
    else {
        render_as_call(op_spelling, inner);
    }
}

void CExprRenderer::visit(AssignmentExpression* e) {
    e->lhs()->accept(this);
    out_ << " = ";
    e->rhs()->accept(this);
}

void CExprRenderer::visit(PowBinaryExpression* e) {
    render_as_call("std::pow", e->lhs(), e->rhs());
}

void CExprRenderer::visit(BinaryExpression *e) {
    static std::unordered_map<tok, const char*> binop_tbl = {
        {tok::minus,    "-"},
        {tok::plus,     "+"},
        {tok::times,    "*"},
        {tok::divide,   "/"},
        {tok::lt,       "<"},
        {tok::lte,      "<="},
        {tok::gt,       ">"},
        {tok::gte,      ">="},
        {tok::equality, "=="}
    };

    if (!binop_tbl.count(e->op())) {
        throw compiler_exception(
            "CExprRender: unsupported binary operator "+token_string(e->op()), e->location());
    }
    const char* op_spelling = binop_tbl.at(e->op());

    auto pop = parent_op_;
    bool use_brackets =
        Lexer::binop_precedence(pop) > Lexer::binop_precedence(e->op())
        || (pop==tok::divide && e->op()==tok::times);
    parent_op_ = e->op();

    auto lhs = e->lhs();
    auto rhs = e->rhs();

    if (use_brackets) out_ << "(";
    lhs->accept(this);
    out_ << op_spelling;
    rhs->accept(this);
    if (use_brackets) out_ << ")";

    // reset parent precedence
    parent_op_ = pop;
}
