#pragma once

#include <sstream>

#include "scope.hpp"
#include "visitor.hpp"

expression_ptr inline_function_call(Expression* e);

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
