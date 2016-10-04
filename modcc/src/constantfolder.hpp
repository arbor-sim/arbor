#pragma once

#include "visitor.hpp"

class ConstantFolderVisitor : public Visitor {
public:
    ConstantFolderVisitor() {}

    void visit(Expression *e)           override;
    // reduce child
    void visit(UnaryExpression *e)      override;
    // reduce left and right children
    void visit(BinaryExpression *e)     override;
    // reduce expressions in arguments
    void visit(NumberExpression *e)     override;

    void visit(CallExpression *e)       override;
    void visit(ProcedureExpression *e)  override;
    void visit(FunctionExpression *e)   override;
    void visit(BlockExpression *e)      override;
    void visit(IfExpression *e)         override;

    // store intermediate results as long double, i.e. 80-bit precision
    long double value = 0.;
    bool  is_number = false;
};

