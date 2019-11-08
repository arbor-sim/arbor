#pragma once

#include <sstream>

#include "scope.hpp"
#include "visitor.hpp"

// Takes an assignment to a function call, returns an inlined
// version without modifying the original expression's contents
expression_ptr inline_function_call(const expression_ptr& e);

class FunctionInliner : public Visitor {

public:

    FunctionInliner(std::string func_name,
                    Expression* lhs,
                    const std::vector<expression_ptr>& fargs,
                    const std::vector<expression_ptr>& cargs,
                    const scope_ptr& scope) :
                    func_name_(func_name), lhs_(lhs->clone()), scope_(scope) {

        for (unsigned i = 0; i < fargs.size(); ++i) {
            call_arg_map_.insert({fargs[i]->is_argument()->spelling(), cargs[i]->clone()});
        }
//        for (auto& c: call_arg_map_) {
//            std::cout << "++ call  [" << c.first << ", " << c.second->to_string() << "]" << std::endl;
//        }

    }

    void visit(Expression* e)            override;
    void visit(UnaryExpression* e)       override;
    void visit(BinaryExpression* e)      override;
    void visit(BlockExpression *e)       override;
    void visit(AssignmentExpression* e)  override;
    void visit(IfExpression* e)          override;
    void visit(LocalDeclaration* e)      override;
    void visit(CallExpression* e)        override;
    void visit(NumberExpression* e)      override {};

    bool return_val_set() {return return_set_;};

    ~FunctionInliner() {}

public:
    std::string func_name_;
    expression_ptr lhs_;
    std::unordered_map<std::string, expression_ptr> call_arg_map_;
    std::unordered_map<std::string, expression_ptr> local_arg_map_;
    scope_ptr scope_;
    bool return_set_ = false;

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
