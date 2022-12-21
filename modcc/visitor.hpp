#pragma once

#include "error.hpp"
#include "expression.hpp"

/// visitor base class
/// The visitors for all AST nodes throw an exception
/// by default, with node types calling the default visitor for a parent
/// For example, all BinaryExpression types call the visitor defined for
/// BinaryExpression, so by overriding just the base implementation, all of the
/// children get that implementation for free, which might be useful for some
/// use cases.
///
/// heavily inspired by the DMD D compiler : github.com/D-Programming-Language/dmd
class Visitor {
public:
    virtual void visit(Expression *e) = 0;
    virtual void visit(Symbol *e)                   { visit((Expression*) e); }
    virtual void visit(LocalVariable *e)            { visit((Expression*) e); }
    virtual void visit(WhiteNoise *e)               { visit((Expression*) e); }
    virtual void visit(IdentifierExpression *e)     { visit((Expression*) e); }
    virtual void visit(NumberExpression *e)         { visit((Expression*) e); }
    virtual void visit(IntegerExpression *e)        { visit((NumberExpression*) e); }
    virtual void visit(LocalDeclaration *e)         { visit((Expression*) e); }
    virtual void visit(ArgumentExpression *e)       { visit((Expression*) e); }
    virtual void visit(PrototypeExpression *e)      { visit((Expression*) e); }
    virtual void visit(CallExpression *e)           { visit((Expression*) e); }
    virtual void visit(ReactionExpression *e)       { visit((Expression*) e); }
    virtual void visit(StoichTermExpression *e)     { visit((Expression*) e); }
    virtual void visit(StoichExpression *e)         { visit((Expression*) e); }
    virtual void visit(CompartmentExpression *e)    { visit((Expression*) e); }
    virtual void visit(VariableExpression *e)       { visit((Expression*) e); }
    virtual void visit(IndexedVariable *e)          { visit((Expression*) e); }
    virtual void visit(FunctionExpression *e)       { visit((Expression*) e); }
    virtual void visit(IfExpression *e)             { visit((Expression*) e); }
    virtual void visit(SolveExpression *e)          { visit((Expression*) e); }
    virtual void visit(DerivativeExpression *e)     { visit((Expression*) e); }
    virtual void visit(PDiffExpression *e)          { visit((Expression*) e); }
    virtual void visit(ProcedureExpression *e)      { visit((Expression*) e); }
    virtual void visit(NetReceiveExpression *e)     { visit((ProcedureExpression*) e); }
    virtual void visit(APIMethod *e)                { visit((Expression*) e); }
    virtual void visit(ConductanceExpression *e)    { visit((Expression*) e); }
    virtual void visit(BlockExpression *e)          { visit((Expression*) e); }
    virtual void visit(InitialBlock *e)             { visit((BlockExpression*) e); }

    virtual void visit(UnaryExpression *e) = 0;
    virtual void visit(NegUnaryExpression *e)       { visit((UnaryExpression*) e); }
    virtual void visit(ExpUnaryExpression *e)       { visit((UnaryExpression*) e); }
    virtual void visit(LogUnaryExpression *e)       { visit((UnaryExpression*) e); }
    virtual void visit(CosUnaryExpression *e)       { visit((UnaryExpression*) e); }
    virtual void visit(SinUnaryExpression *e)       { visit((UnaryExpression*) e); }
    virtual void visit(SqrtUnaryExpression *e)      { visit((UnaryExpression*) e); }
    virtual void visit(StepRightUnaryExpression *e) { visit((UnaryExpression*) e); }
    virtual void visit(StepLeftUnaryExpression *e)  { visit((UnaryExpression*) e); }
    virtual void visit(StepUnaryExpression *e)      { visit((UnaryExpression*) e); }
    virtual void visit(SignumUnaryExpression *e)    { visit((UnaryExpression*) e); }

    virtual void visit(BinaryExpression *e) = 0;
    virtual void visit(ConditionalExpression *e)    { visit((BinaryExpression*) e); }
    virtual void visit(AssignmentExpression *e)     { visit((BinaryExpression*) e); }
    virtual void visit(ConserveExpression *e)       { visit((BinaryExpression*) e); }
    virtual void visit(LinearExpression *e)         { visit((BinaryExpression*) e); }
    virtual void visit(AddBinaryExpression *e)      { visit((BinaryExpression*) e); }
    virtual void visit(SubBinaryExpression *e)      { visit((BinaryExpression*) e); }
    virtual void visit(MulBinaryExpression *e)      { visit((BinaryExpression*) e); }
    virtual void visit(DivBinaryExpression *e)      { visit((BinaryExpression*) e); }
    virtual void visit(PowBinaryExpression *e)      { visit((BinaryExpression*) e); }

    virtual ~Visitor() {};
};

// Visitor specialization intended for use as a base class for visitors that
// operate as function or procedure body rewriters after semantic analysis.
//
// Block rewriter visitors construct a new block body from a supplied
// `BlockExpression`, `ProcedureExpression` or `FunctionExpression`. By default,
// expressions are simply copied to the list of statements corresponding to the
// rewritten block; nested blocks as provided by `IfExpression` objects are
// handled recursively.
//
// The `finalize()` method is called after visiting all the statements in the
// top-level block, and is intended to be overrided by derived classes as required.
//
// The `as_block()` method is intended to be called by users of the derived block
// rewriter objects. It constructs a new `BlockExpression` from the accumulated
// replacement statements and applies a semantic pass if the `BlockRewriterBase`
// was given a corresponding scope
//
// The visitor maintains significant internal state: the `reset` method should
// be called between visits of top-level blocks.
//
// Errors are recorded through the `error_stack` mixin rather than by
// throwing an exception.

class BlockRewriterBase : public Visitor, public error_stack {
public:
    BlockRewriterBase() {}
    BlockRewriterBase(scope_ptr block_scope):
        block_scope_(block_scope) {}

    virtual void visit(Expression *e) override {
        statements_.push_back(e->clone());
    }

    virtual void visit(UnaryExpression *e) override { visit((Expression*)e); }
    virtual void visit(BinaryExpression *e) override { visit((Expression*)e); }

    virtual void visit(BlockExpression *e) override {
        bool top = !started_;
        if (top) {
            loc_ = e->location();
            started_ = true;

            if (!block_scope_) {
                block_scope_ = e->scope();
            }
        }

        for (auto& s: e->statements()) {
            s->accept(this);
        }

        if (top) {
            finalize();
        }
    }

    virtual void visit(IfExpression* e) override {
        expr_list_type outer;
        std::swap(outer, statements_);

        e->true_branch()->accept(this);
        auto true_branch = make_expression<BlockExpression>(
            e->true_branch()->location(),
            std::move(statements_),
            true);

        statements_.clear();
        expression_ptr false_branch;
        if (e->false_branch()) {
            e->false_branch()->accept(this);
            false_branch = make_expression<BlockExpression>(
                    e->false_branch()->location(),
                    std::move(statements_),
                    true);
        }

        statements_ = std::move(outer);
        statements_.push_back(make_expression<IfExpression>(
            e->location(),
            e->condition()->clone(),
            std::move(true_branch),
            std::move(false_branch)));
    }

    virtual void visit(ProcedureExpression* e) override {
        e->body()->accept(this);
    }

    virtual void visit(FunctionExpression* e) override {
        e->body()->accept(this);
    }

    virtual expression_ptr as_block(bool is_nested=false) {
        if (has_error()) return nullptr;

        expr_list_type body_stmts;
        for (const auto& s: statements_) body_stmts.push_back(s->clone());

        auto body = make_expression<BlockExpression>(
            loc_,
            std::move(body_stmts),
            is_nested);

        if (block_scope_) {
            body->semantic(block_scope_);
        }
        return body;
    }

    // Reset state.
    virtual void reset() {
        statements_.clear();
        started_ = false;
        loc_ = Location{};
        clear_errors();
        clear_warnings();
    }

protected:
    // False until processing of top block starts.
    bool started_ = false;

    // Location of original block.
    Location loc_;

    // Scope for semantic pass.
    scope_ptr block_scope_;

    // Statements in replacement procedure body.
    expr_list_type statements_;

    // Finalise statements list at end of top block visit.
    virtual void finalize() {}
};

