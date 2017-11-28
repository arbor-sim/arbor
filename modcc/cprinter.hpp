#pragma once

#include <sstream>

#include "module.hpp"
#include "textbuffer.hpp"
#include "visitor.hpp"

class CPrinter: public Visitor {
public:
    CPrinter() {}
    explicit CPrinter(Module &m): module_(&m) {}

    virtual void visit(Expression *e)           override;
    virtual void visit(UnaryExpression *e)      override;
    virtual void visit(BinaryExpression *e)     override;
    virtual void visit(NumberExpression *e)     override;
    virtual void visit(VariableExpression *e)   override;
    virtual void visit(Symbol *e)               override;
    virtual void visit(LocalVariable *e)        override;
    virtual void visit(IndexedVariable *e)      override;
    virtual void visit(CellIndexedVariable *e)  override;
    virtual void visit(IdentifierExpression *e) override;
    virtual void visit(CallExpression *e)       override;
    virtual void visit(ProcedureExpression *e)  override;
    virtual void visit(APIMethod *e)            override;
    virtual void visit(LocalDeclaration *e)     override;
    virtual void visit(BlockExpression *e)      override;
    virtual void visit(IfExpression *e)         override;

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
    void clear_text() {
        text_.clear();
    }

    virtual ~CPrinter() { }

    virtual std::string emit_source();
    virtual void emit_headers();
    virtual void emit_api_loop(APIMethod* e,
                               const std::string& start,
                               const std::string& end,
                               const std::string& inc);

protected:
    void print_mechanism(Visitor *backend);
    void print_APIMethod(APIMethod* e);

    Module *module_ = nullptr;
    TextBuffer text_;
    bool aliased_output_ = false;

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

    bool is_arg_local(Symbol *s) {
        if(auto l=s->is_local_variable()) {
            if(l->is_arg()) {
                return true;
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

    bool is_ghost_local(Symbol *s) {
        if(!is_point_process()) return false;
        if(!aliased_output_)    return false;
        if(is_arg_local(s))     return false;
        return is_output(s);
    }

    bool is_stack_local(Symbol *s) {
        if(is_arg_local(s))    return false;
        return !is_ghost_local(s);
    }

    bool is_point_process() {
        return module_ && module_->kind() == moduleKind::point;
    }

    std::vector<LocalVariable*> aliased_vars(APIMethod* e);
};
