#pragma once

#include <sstream>

#include "module.hpp"
#include "textbuffer.hpp"
#include "visitor.hpp"

class CUDAPrinter : public Visitor {
public:
    CUDAPrinter() {}
    CUDAPrinter(Module &m, bool o=false);

    void visit(Expression *e)           override;
    void visit(UnaryExpression *e)      override;
    void visit(BinaryExpression *e)     override;
    void visit(AssignmentExpression *e) override;
    void visit(PowBinaryExpression *e)  override;
    void visit(NumberExpression *e)     override;
    void visit(VariableExpression *e)   override;

    void visit(Symbol *e)               override;
    void visit(LocalVariable *e)        override;
    void visit(IndexedVariable *e)      override;

    void visit(IdentifierExpression *e) override;
    void visit(CallExpression *e)       override;
    void visit(ProcedureExpression *e)  override;
    void visit(APIMethod *e)            override;
    void visit(LocalDeclaration *e)      override;
    void visit(BlockExpression *e)      override;
    void visit(IfExpression *e)         override;

    std::string text() const {
        return text_.str();
    }

    void set_gutter(int w) {
        text_.set_gutter(w);
    }
    void increase_indentation(){
        text_.increase_indentation();
    }
    void decrease_indentation(){
        text_.decrease_indentation();
    }
private:

    bool is_input(Symbol *s) {
        if(auto l = s->is_local_variable() ) {
            if(l->is_local()) {
                if(l->is_indexed() && l->is_read()) {
                    return true;
                }
            }
        }
        return false;
    }

    bool is_output(Symbol *s) {
        if(auto l = s->is_local_variable() ) {
            if(l->is_local()) {
                if(l->is_indexed() && l->is_write()) {
                    return true;
                }
            }
        }
        return false;
    }

    bool is_indexed_local(Symbol *s) {
        if(auto l=s->is_local_variable()) {
            if(l->is_indexed()) {
                return true;
            }
        }
        return false;
    }

    bool is_arg_local(Symbol *s) {
        if(auto l=s->is_local_variable()) {
            if(l->is_arg()) {
                return true;
            }
        }
        return false;
    }

    bool is_stack_local(Symbol *s) {
        if(is_arg_local(s))    return false;
        if(is_input(s))        return false;
        if(is_output(s))       return false;
        return true;
    }

    bool is_point_process() const {
        return module_->kind() == moduleKind::point;
    }

    void print_APIMethod_body(APIMethod* e);
    void print_procedure_prototype(ProcedureExpression *e);
    std::string index_string(Symbol *e);

    Module *module_ = nullptr;
    tok parent_op_ = tok::eq;
    TextBuffer text_;
    //bool optimize_ = false;
};

