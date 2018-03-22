#pragma once

#include <iosfwd>
#include <string>

#include "cexpr_emit.hpp"
#include "module.hpp"
#include "visitor.hpp"

std::string emit_cpp_source(const Module& m, const std::string& ns);

// CPrinter class exposed in header for testing purposes only (but maybe also for SIMD printer?!)

class CPrinter2: public Visitor {
public:
    CPrinter2(std::ostream& out): out_(out) {}

    void visit(Expression* e) override {
        throw compiler_exception("CPrinter cannot translate expression "+e->to_string());
    }

    void visit(BlockExpression*) override;
    void visit(CallExpression*) override;
    void visit(IdentifierExpression*) override;
    void visit(VariableExpression*) override;
    void visit(LocalVariable*) override;
    void visit(IndexedVariable*) override;

    // Delegate low-level emits to cexpr_emit:
    void visit(NumberExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(UnaryExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(BinaryExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(IfExpression* e) override { cexpr_emit(e, out_, this); }

private:
    std::ostream& out_;
};



