#pragma once

#include <iosfwd>
#include <string>

#include "module.hpp"
#include "visitor.hpp"

#include "printer/cexpr_emit.hpp"
#include "printer/simd.hpp"

std::string emit_cpp_source(const Module& m, const std::string& ns, simd_spec simd);

// CPrinter and SimdPrinter visitors exposed in header for testing purposes only.

class CPrinter: public Visitor {
public:
    CPrinter(std::ostream& out): out_(out) {}

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

protected:
    std::ostream& out_;
};

class SimdPrinter: public Visitor {
public:
    SimdPrinter(std::ostream& out): out_(out) {}
    SimdPrinter(std::ostream& out, bool is_indexed): out_(out), is_indexed_(is_indexed) {}

    void visit(Expression* e) override {
        throw compiler_exception("SimdPrinter cannot translate expression "+e->to_string());
    }

    void visit(BlockExpression*) override;
    void visit(CallExpression*) override;
    void visit(IdentifierExpression*) override;
    void visit(VariableExpression*) override;
    void visit(LocalVariable*) override;
    void visit(IndexedVariable*) override;
    void visit(AssignmentExpression*) override;

    void visit(NumberExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(UnaryExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(BinaryExpression* e) override { cexpr_emit(e, out_, this); }

private:
    std::ostream& out_;
    bool is_indexed_;
};
