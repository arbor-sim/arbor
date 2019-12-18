#pragma once

#include <sstream>

#include "scope.hpp"
#include "visitor.hpp"

expression_ptr inline_function_calls(BlockExpression* block, std::string calling_func);

class FunctionInliner : public BlockRewriterBase {
public:
    using BlockRewriterBase::visit;

    FunctionInliner(std::string calling_func) : BlockRewriterBase(), calling_func_(calling_func) {};
    FunctionInliner(scope_ptr s): BlockRewriterBase(s) {}

    virtual void visit(Expression *e)            override;
    virtual void visit(CallExpression *e)        override;
    virtual void visit(ConserveExpression *e)    override;
    virtual void visit(CompartmentExpression *e) override;
    virtual void visit(LinearExpression *e)      override;
    virtual void visit(AssignmentExpression* e)  override;
    virtual void visit(BinaryExpression* e)      override;
    virtual void visit(UnaryExpression* e)       override;
    virtual void visit(IfExpression* e)          override;
    virtual void visit(LocalDeclaration* e)      override;
    virtual void visit(NumberExpression* e)      override {};
    virtual void visit(IdentifierExpression* e)  override {};

    bool return_val_set() {return return_set_;};
    bool finished_inlining() {return !inlining_executed_;};

    ~FunctionInliner() {}

private:
    std::string func_name_, calling_func_;
    expression_ptr lhs_;
    std::unordered_map<std::string, expression_ptr> call_arg_map_;
    std::unordered_map<std::string, expression_ptr> local_arg_map_;
    scope_ptr scope_;

    // Tracks whether the return value of a function has been set
    bool return_set_ = true;

    // Tracks whether a function is being inlined
    bool inlining_in_progress_ = false;

    // Tracks whether a function has been inlined
    bool inlining_executed_ = false;

    void replace_args(Expression* e);

protected:
    virtual void reset() override {
        func_name_.clear();
        lhs_ = nullptr;
        call_arg_map_.clear();
        local_arg_map_.clear();
        scope_.reset();
        return_set_ = true;
        inlining_in_progress_ = false;
        inlining_executed_ = false;
        BlockRewriterBase::reset();
    }
};

class VariableReplacer : public Visitor {

public:

    VariableReplacer(std::string const& source, std::string const& target)
    :   source_(source),
        target_(target)
    {}

    void visit(Expression *e)           override;
    void visit(UnaryExpression *e)      override;
    void visit(BinaryExpression *e)     override;
    void visit(NumberExpression *e)     override {};

    ~VariableReplacer() {}

private:

    std::string source_;
    std::string target_;
};

class ValueInliner : public Visitor {

public:

    ValueInliner(std::string const& source, long double value)
    :   source_(source),
        value_(value)
    {}

    void visit(Expression *e)           override;
    void visit(UnaryExpression *e)      override;
    void visit(BinaryExpression *e)     override;
    void visit(NumberExpression *e)     override {};

    ~ValueInliner() {}

private:

    std::string source_;
    long double value_;
};
