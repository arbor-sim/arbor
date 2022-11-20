#pragma once

#include <cstdio>
#include <iomanip>
#include <set>

#include "visitor.hpp"

struct FlopAccumulator {
    int add=0;
    int neg=0;
    int mul=0;
    int div=0;
    int exp=0;
    int sin=0;
    int cos=0;
    int log=0;
    int pow=0;
    int sqrt=0;

    void reset() {
        add = neg = mul = div = exp = sin = cos = log = pow = sqrt = 0;
    }
};

static std::ostream& operator << (std::ostream& os, FlopAccumulator const& f) {
    char buffer[512];
    snprintf(buffer,
             512,
             "   add   neg   mul   div   exp   sin   cos   log   pow  sqrt\n%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d",
             f.add, f.neg, f.mul, f.div, f.exp, f.sin, f.cos, f.log, f.pow, f.sqrt);

    os << buffer << std::endl << std::endl;
    os << " add+mul+neg  " << f.add + f.neg + f.mul << std::endl;
    os << " div          " << f.div << std::endl;
    os << " exp          " << f.exp;

    return os;
}

class FlopVisitor : public Visitor {
public:
    void visit(Expression *e) override {}

    // traverse the statements in an API method
    void visit(APIMethod *e) override {
        for(auto& expression : *(e->body())) {
            expression->accept(this);
        }
    }

    // traverse the statements in a procedure
    void visit(ProcedureExpression *e) override {
        for(auto& expression : *(e->body())) {
            expression->accept(this);
        }
    }

    // traverse the statements in a function
    void visit(FunctionExpression *e) override {
        for(auto& expression : *(e->body())) {
            expression->accept(this);
        }
    }

    ////////////////////////////////////////////////////
    // specializations for each type of unary expression
    // leave UnaryExpression to throw, to catch
    // any missed specializations
    ////////////////////////////////////////////////////
    void visit(UnaryExpression *e) override {
        // do nothing
    }

    void visit(NegUnaryExpression *e) override {
        // this is a simplification
        // we would have to perform analysis of parent nodes to ensure that
        // the negation actually translates into an operation
        //  :: x - - x      // not counted
        //  :: x * -exp(3)  // should be counted
        //  :: x / -exp(3)  // should be counted
        //  :: x / - -exp(3)// should be counted only once
        flops.neg++;
    }
    void visit(ExpUnaryExpression *e) override {
        e->expression()->accept(this);
        flops.exp++;
    }
    void visit(LogUnaryExpression *e) override {
        e->expression()->accept(this);
        flops.log++;
    }
    void visit(CosUnaryExpression *e) override {
        e->expression()->accept(this);
        flops.cos++;
    }
    void visit(SinUnaryExpression *e) override {
        e->expression()->accept(this);
        flops.sin++;
    }
    void visit(SqrtUnaryExpression *e) override {
        e->expression()->accept(this);
        flops.sqrt++;
    }
    void visit(StepUnaryExpression *e) override {
        e->expression()->accept(this);
        flops.add+=2;
        flops.mul++;
    }
    void visit(SignumUnaryExpression *e) override {
        e->expression()->accept(this);
        flops.add++;
    }

    ////////////////////////////////////////////////////
    // specializations for each type of binary expression
    // leave UnaryExpression throw an exception, to catch
    // any missed specializations
    ////////////////////////////////////////////////////
    void visit(BinaryExpression *e) override {
        // there must be a specialization of the flops counter for every type
        // of binary expression: if we get here there has been an attempt to
        // visit a binary expression for which no visitor is implemented
        throw compiler_exception(
            "PerfVisitor unable to analyse binary expression " + e->to_string(),
            e->location());
    }
    void visit(AssignmentExpression *e) override {
        e->rhs()->accept(this);
    }
    void visit(AddBinaryExpression *e)  override {
        e->lhs()->accept(this);
        e->rhs()->accept(this);
        flops.add++;
    }
    void visit(SubBinaryExpression *e)  override {
        e->lhs()->accept(this);
        e->rhs()->accept(this);
        flops.add++;
    }
    void visit(MulBinaryExpression *e)  override {
        e->lhs()->accept(this);
        e->rhs()->accept(this);
        flops.mul++;
    }
    void visit(DivBinaryExpression *e)  override {
        e->lhs()->accept(this);
        e->rhs()->accept(this);
        flops.div++;
    }
    void visit(PowBinaryExpression *e)  override {
        e->lhs()->accept(this);
        e->rhs()->accept(this);
        flops.pow++;
    }

    FlopAccumulator flops;

    std::string print() const {
        std::stringstream s;

        s << flops << std::endl;

        return s.str();
    }
};

class MemOpVisitor : public Visitor {
public:
    void visit(Expression *e) override {}

    // traverse the statements in an API method
    void visit(APIMethod *e) override {
        for(auto& expression : *(e->body())) {
            expression->accept(this);
        }

        // create local indexed views
        for(auto &symbol : e->scope()->locals()) {
            auto var = symbol.second->is_local_variable();
            if(var->is_indexed()) {
                if(var->is_read()) {
                    indexed_reads_.insert(var);
                }
                else {
                    indexed_writes_.insert(var);
                }
            }
        }
    }

    // traverse the statements in a procedure
    void visit(ProcedureExpression *e) override {
        for(auto& expression : *(e->body())) {
            expression->accept(this);
        }
    }

    // traverse the statements in a function
    void visit(FunctionExpression *e) override {
        for(auto& expression : *(e->body())) {
            expression->accept(this);
        }
    }

    void visit(UnaryExpression *e) override {
        e->expression()->accept(this);
    }

    void visit(BinaryExpression *e) override {
        e->lhs()->accept(this);
        e->rhs()->accept(this);
    }

    void visit(AssignmentExpression *e) override {
        // handle the write on the lhs as a special case
        auto symbol = e->lhs()->is_identifier()->symbol();
        if(!symbol) {
            throw compiler_exception(
                " attempt to look up name of identifier for which no symbol_ yet defined",
                e->lhs()->location());
        }
        switch (symbol->kind()) {
            case symbolKind::variable :
                vector_writes_.insert(symbol);
                break;
            case symbolKind::indexed_variable :
                indexed_writes_.insert(symbol);
            default :
                break;
        }

        // let the visitor implementation handle the reads
        e->rhs()->accept(this);
    }

    void visit(IdentifierExpression* e) override {
        auto symbol = e->symbol();
        if(!symbol) {
            throw compiler_exception(
                " attempt to look up name of identifier for which no symbol_ yet defined",
                e->location());
        }
        switch (symbol->kind()) {
            case symbolKind::variable :
                if(symbol->is_variable()->is_range()) {
                    vector_reads_.insert(symbol);
                }
                break;
            case symbolKind::indexed_variable :
                indexed_reads_.insert(symbol);
            default :
                break;
        }
    }

    std::string print() const {
        std::stringstream s;

        auto ir = indexed_reads_.size();
        auto vr = vector_reads_.size();
        auto iw = indexed_writes_.size();
        auto vw = vector_writes_.size();

        auto w = std::setw(8);
        s << "        " << w << "read" << w << "write" << w << "total" << std::endl;
        s << "indexed " << w << ir     << w << iw      << w << ir + iw << std::endl;
        s << "vector  " << w << vr     << w << vw      << w << vr + vw << std::endl;
        s << "total   " << w << vr+ir  << w << vw +iw  << w << vr + vw + ir +iw << std::endl;

        return s.str();
    }

private:
    std::set<Symbol*> indexed_reads_;
    std::set<Symbol*> vector_reads_;
    std::set<Symbol*> indexed_writes_;
    std::set<Symbol*> vector_writes_;
};
