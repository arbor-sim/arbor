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
    void visit(NumberExpression *e)     override;
    void visit(VariableExpression *e)   override;

    void visit(Symbol *e)               override;
    void visit(LocalVariable *e)        override;
    void visit(IndexedVariable *e)      override;
    void visit(CellIndexedVariable *e)  override;

    void visit(IdentifierExpression *e) override;
    void visit(CallExpression *e)       override;
    void visit(ProcedureExpression *e)  override;
    void visit(APIMethod *e)            override;
    void visit(LocalDeclaration *e)     override;
    void visit(BlockExpression *e)      override;
    void visit(IfExpression *e)         override;

    std::string impl_header_text() const {
        return impl_interface_.str();
    }

    std::string impl_text() const {
        return impl_.str();
    }

    std::string interface_text() const {
        return interface_.str();
    }

    // public for testing purposes:
    void set_buffer(TextBuffer& buf) {
        current_buffer_ = &buf;
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

    void print_APIMethod_body(ProcedureExpression* e);
    std::string APIMethod_prototype(APIMethod *e);
    std::string pack_name();
    void print_device_function_prototype(ProcedureExpression *e);
    std::string index_string(Symbol *e);

    std::string module_name_;
    Module *module_ = nullptr;

    TextBuffer interface_;
    TextBuffer impl_;
    TextBuffer impl_interface_;
    TextBuffer* current_buffer_;

    TextBuffer& buffer() {
        if (!current_buffer_) {
            throw std::runtime_error("CUDAPrinter buffer must be set via CUDAPrinter::set_buffer() before accessing via CUDAPrinter::buffer().");
        }
        return *current_buffer_;
    }
};

