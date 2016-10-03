#pragma once

#include "error.hpp"
#include "expression.hpp"
#include "util.hpp"

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
    virtual void visit(Symbol *e)               { visit((Expression*) e); }
    virtual void visit(LocalVariable *e)        { visit((Expression*) e); }
    virtual void visit(IdentifierExpression *e) { visit((Expression*) e); }
    virtual void visit(NumberExpression *e)     { visit((Expression*) e); }
    virtual void visit(LocalDeclaration *e)     { visit((Expression*) e); }
    virtual void visit(ArgumentExpression *e)   { visit((Expression*) e); }
    virtual void visit(PrototypeExpression *e)  { visit((Expression*) e); }
    virtual void visit(CallExpression *e)       { visit((Expression*) e); }
    virtual void visit(VariableExpression *e)   { visit((Expression*) e); }
    virtual void visit(IndexedVariable *e)      { visit((Expression*) e); }
    virtual void visit(FunctionExpression *e)   { visit((Expression*) e); }
    virtual void visit(IfExpression *e)         { visit((Expression*) e); }
    virtual void visit(SolveExpression *e)      { visit((Expression*) e); }
    virtual void visit(DerivativeExpression *e) { visit((Expression*) e); }
    virtual void visit(ProcedureExpression *e)  { visit((Expression*) e); }
    virtual void visit(NetReceiveExpression *e) { visit((ProcedureExpression*) e); }
    virtual void visit(APIMethod *e)            { visit((Expression*) e); }
    virtual void visit(BlockExpression *e)      { visit((Expression*) e); }
    virtual void visit(InitialBlock *e)         { visit((BlockExpression*) e); }

    virtual void visit(UnaryExpression *e) = 0;
    virtual void visit(NegUnaryExpression *e)   { visit((UnaryExpression*) e); }
    virtual void visit(ExpUnaryExpression *e)   { visit((UnaryExpression*) e); }
    virtual void visit(LogUnaryExpression *e)   { visit((UnaryExpression*) e); }
    virtual void visit(CosUnaryExpression *e)   { visit((UnaryExpression*) e); }
    virtual void visit(SinUnaryExpression *e)   { visit((UnaryExpression*) e); }

    virtual void visit(BinaryExpression *e) = 0;
    virtual void visit(AssignmentExpression *e) { visit((BinaryExpression*) e); }
    virtual void visit(AddBinaryExpression *e)  { visit((BinaryExpression*) e); }
    virtual void visit(SubBinaryExpression *e)  { visit((BinaryExpression*) e); }
    virtual void visit(MulBinaryExpression *e)  { visit((BinaryExpression*) e); }
    virtual void visit(DivBinaryExpression *e)  { visit((BinaryExpression*) e); }
    virtual void visit(PowBinaryExpression *e)  { visit((BinaryExpression*) e); }

    virtual ~Visitor() {};
};
