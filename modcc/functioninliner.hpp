#pragma once

#include <sstream>

#include "scope.hpp"
#include "visitor.hpp"

expression_ptr inline_function_calls(BlockExpression* block);

class FunctionInliner : public BlockRewriterBase {
public:
    using BlockRewriterBase::visit;

    FunctionInliner() : BlockRewriterBase() {};
    FunctionInliner(scope_ptr s): BlockRewriterBase(s) {}

    virtual void visit(Expression *e)            override;
    virtual void visit(CallExpression *e)        override;
    virtual void visit(AssignmentExpression* e)  override;
    virtual void visit(BinaryExpression* e)      override;
    virtual void visit(UnaryExpression* e)       override;
    virtual void visit(IfExpression* e)          override;
    virtual void visit(LocalDeclaration* e)      override;
    virtual void visit(NumberExpression* e)      override {};
    virtual void visit(IdentifierExpression* e)  override {};

    bool return_val_set() {return return_set_;};
    bool still_inlining() {return inlined_func_;};

    ~FunctionInliner() {}

protected:
    virtual void reset() override {
        BlockRewriterBase::reset();
    }

private:
    std::string func_name_;
    expression_ptr lhs_;
    std::unordered_map<std::string, expression_ptr> call_arg_map_;
    std::unordered_map<std::string, expression_ptr> local_arg_map_;
    scope_ptr scope_;
    bool return_set_ = false;
    bool processing_function_call_ = false;
    bool inlined_func_ = false;

    void replace_args(Expression* e);

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
