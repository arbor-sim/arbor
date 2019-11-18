#pragma once

#include <iosfwd>
#include <unordered_set>

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

class SimdIfEmitter: public CExprEmitter {
    using CExprEmitter::visit;
public:
    SimdIfEmitter(std::ostream& out, bool input_mask, bool is_indirect, Visitor* fallback): CExprEmitter(out, fallback) {
        if (input_mask) {
            current_mask_ = "mask_input_";
            current_mask_bar_ = "!mask_input";
        }
    }

    void visit(BlockExpression *e) override;
    void visit(CallExpression *e) override;
    void visit(AssignmentExpression *e) override;
    void visit(IfExpression *e) override;

protected:
    static std::unordered_set<std::string> mask_names_;
    std::string current_mask_, current_mask_bar_;
    bool processing_true_;
    bool is_indirect_;

private:
    std::string make_unique_var(scope_ptr scope, std::string prefix) {
        for (int i = 0;; ++i) {
            std::string name = prefix + std::to_string(i) + "_";
            if (!scope->find(name) && !mask_names_.count(name)) {
                mask_names_.insert(name);
                return name;
            }
        }
    };
};

inline void simd_if_emit(Expression* e, std::ostream& out, bool is_masked, bool is_indirect, Visitor* fallback) {
    SimdIfEmitter emitter(out, is_masked, is_indirect, fallback);
    e->accept(&emitter);
}

// Helper for formatting of double-valued numeric constants.
struct as_c_double {
    double value;
    as_c_double(double value): value(value) {}
};

std::ostream& operator<<(std::ostream&, as_c_double);
