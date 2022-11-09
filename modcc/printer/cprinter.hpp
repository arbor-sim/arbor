#pragma once

#include <iosfwd>
#include <string>

#include "module.hpp"
#include "visitor.hpp"
#include <libmodcc/export.hpp>

#include "printer/cexpr_emit.hpp"
#include "printer/printeropt.hpp"

ARB_LIBMODCC_API std::string emit_cpp_source(const Module& m, const printer_options& opt);

// CPrinter and SimdPrinter visitors exposed in header for testing purposes only.

class ARB_LIBMODCC_API CPrinter: public Visitor {
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
    void visit(WhiteNoise*) override;

    // Delegate low-level emits to cexpr_emit:
    void visit(NumberExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(UnaryExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(BinaryExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(IfExpression* e) override { cexpr_emit(e, out_, this); }

protected:
    std::ostream& out_;
};


enum class simd_expr_constraint{
    constant,
    contiguous,
    other
};

struct ApiFlags {
    bool cv_loop = true;
    bool ppack_iface=true;
    bool use_additive=false;
    bool is_point=false;
    bool can_write_voltage=false;

    ApiFlags& loop(bool v) { cv_loop = v; return *this; }
    ApiFlags& iface(bool v) { ppack_iface = v; return *this; }
    ApiFlags& additive(bool v) { use_additive = v; return *this; }
    ApiFlags& point(bool v) { is_point = v; return *this; }
    ApiFlags& voltage(bool v) { can_write_voltage = v; return *this; }
};

const ApiFlags net_recv_flags = {false, false, true}; // No CV loop, no PPACK, use additive
const ApiFlags post_evt_flags = {false, false};       // No CV loop, no PPACK

class ARB_LIBMODCC_API SimdPrinter: public Visitor {
public:
    SimdPrinter(std::ostream& out): out_(out) {}

    void visit(Expression* e) override {
        throw compiler_exception("SimdPrinter cannot translate expression "+e->to_string());
    }
    void set_var_indexed(bool is_indirect_index) {
        is_indirect_ = is_indirect_index;
    }
    void set_input_mask(std::string input_mask) {
        input_mask_ = input_mask;
    }
    void save_scalar_names(const std::unordered_set<std::string>& scalars) {
        scalars_ = scalars;
    }

    void visit(BlockExpression*) override;
    void visit(CallExpression*) override;
    void visit(IdentifierExpression*) override;
    void visit(VariableExpression*) override;
    void visit(LocalVariable*) override;
    void visit(AssignmentExpression*) override;
    void visit(WhiteNoise*) override;

    void visit(NumberExpression* e) override { simd_expr_emit(e, out_, is_indirect_, input_mask_, scalars_, this); } 
    void visit(UnaryExpression* e)  override { simd_expr_emit(e, out_, is_indirect_, input_mask_, scalars_, this); }
    void visit(BinaryExpression* e) override { simd_expr_emit(e, out_, is_indirect_, input_mask_, scalars_, this); }
    void visit(IfExpression* e)     override { simd_expr_emit(e, out_, is_indirect_, input_mask_, scalars_, this); }

private:
    std::ostream& out_;
    std::string input_mask_;
    bool is_indirect_ = false; // For choosing between "index_" and "i_" as an index. Depends on whether
                               // we are in a procedure or handling a simd constraint in an API call.
    std::unordered_set<std::string> scalars_;
};
