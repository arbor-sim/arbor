#pragma once

#include "expression.hpp"
#include "visitor.hpp"

// Common functionality for generating source from binary expressions
// as C expressions.

class CExprEmitter: public Visitor {
public:
    CExprEmitter(std::ostream& out, Visitor* fallback):
        out_(out), fallback_(fallback)
    {}

    void visit(Expression* e) override { e->accept(fallback_); }

    void visit(UnaryExpression *e)      override;
    void visit(BinaryExpression *e)     override;
    void visit(AssignmentExpression *e) override;
    void visit(PowBinaryExpression *e)  override;
    void visit(NumberExpression *e)     override;

protected:
    std::ostream& out_;
    Visitor* fallback_;

    void emit_as_call(const char* sub, Expression*);
    void emit_as_call(const char* sub, Expression*, Expression*);
};

inline void cexpr_emit(Expression* e, std::ostream& out, Visitor* fallback) {
    CExprEmitter renderer(out, fallback);
    e->accept(&renderer);
}
