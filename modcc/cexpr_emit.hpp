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

    virtual void visit(Expression* e) { e->accept(fallback_); }

    virtual void visit(UnaryExpression *e)      override;
    virtual void visit(BinaryExpression *e)     override;
    virtual void visit(AssignmentExpression *e) override;
    virtual void visit(PowBinaryExpression *e)  override;
    virtual void visit(NumberExpression *e)     override;

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
