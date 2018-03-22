#pragma once

#include <iosfwd>

#include "expression.hpp"
#include "visitor.hpp"

// Common functionality for generating source from binary expressions
// and conditional structures with C syntax.

class CExprEmitter: public Visitor {
public:
    CExprEmitter(std::ostream& out, Visitor* fallback):
        out_(out), fallback_(fallback)
    {}

    void visit(Expression* e) override { e->accept(fallback_); }

    void visit(UnaryExpression *e) override;
    void visit(BinaryExpression *e) override;
    void visit(AssignmentExpression *e) override;
    void visit(PowBinaryExpression *e) override;
    void visit(NumberExpression *e) override;
    void visit(IfExpression *e) override;

protected:
    std::ostream& out_;
    Visitor* fallback_;

    void emit_as_call(const char* sub, Expression*);
    void emit_as_call(const char* sub, Expression*, Expression*);
};

inline void cexpr_emit(Expression* e, std::ostream& out, Visitor* fallback) {
    CExprEmitter emitter(out, fallback);
    e->accept(&emitter);
}

// Helper for formatting of double-valued numeric constants.
struct as_c_double {
    double value;
    as_c_double(double value): value(value) {}
};

std::ostream& operator<<(std::ostream&, as_c_double);
