#pragma once

#include <set>
#include <sstream>

#include "backends/simd.hpp"
#include "cprinter.hpp"
#include "modccutil.hpp"
#include "options.hpp"
#include "textbuffer.hpp"


using namespace nest::mc;

template<targetKind Arch>
class SimdPrinter : public CPrinter {
public:
    SimdPrinter()
        : cprinter_(make_unique<CPrinter>())
    {}

    // Initialize our base CPrinter in default unoptimized mode; we handle the
    // vectorization ourselves
    SimdPrinter(Module& m, bool optimize = false)
        : CPrinter(m),
          cprinter_(make_unique<CPrinter>(m))
    { }

    void visit(NumberExpression *e) override {
        simd_backend::emit_set_value(text_, e->value());
    }

    void visit(UnaryExpression *e) override;
    void visit(BinaryExpression *e) override;
    void visit(PowBinaryExpression *e) override;
    void visit(ProcedureExpression *e) override;
    void visit(AssignmentExpression *e) override;
    void visit(VariableExpression *e) override;
    void visit(LocalVariable *e) override {
        const std::string& name = e->name();
        text_ << name;
    }

    void visit(CellIndexedVariable *e) override;
    void visit(IndexedVariable *e) override;
    void visit(APIMethod *e) override;
    void visit(BlockExpression *e) override;
    void visit(CallExpression *e) override;

    void emit_headers() override {
        CPrinter::emit_headers();
        text_.add_line("#include <climits>");
        text_ << simd_backend::emit_headers();
        text_.add_line();
    }

    void emit_api_loop(APIMethod* e,
                       const std::string& start,
                       const std::string& end,
                       const std::string& inc) override;

private:
    using simd_backend = modcc::simd_intrinsics<Arch>;

    void emit_indexed_view(LocalVariable* var, std::set<std::string>& decls);
    void emit_indexed_view_simd(LocalVariable* var, std::set<std::string>& decls);

    // variable naming conventions
    std::string emit_member_name(const std::string& varname) {
        return varname + "_";
    }


    std::string emit_rawptr_name(const std::string& varname) {
        return "r_" + varname;
    }

    std::pair<std::string, std::string>
    emit_rawptr_ion(const std::string& iname, const std::string& ifield) {
        return std::make_pair(emit_rawptr_name(iname),
                              emit_rawptr_name(iname + "_" + ifield));
    }

    std::string emit_vindex_name(const std::string& varname) {
        return "v_" + varname + "_index";
    }

    std::string emit_vtmp_name(const std::string& varname) {
        return "v_" + varname;
    }

    // CPrinter to delegate generation of unvectorised code
    std::unique_ptr<CPrinter> cprinter_;

    // Treat range access as loads
    bool range_load_ = true;
};

template<targetKind Arch>
void SimdPrinter<Arch>::visit(APIMethod *e) {
    text_.add_gutter() << "void " << e->name() << "() override {\n";
    if (!e->scope()) { // error: semantic analysis has not been performed
        throw compiler_exception(
            "SimdPrinter attempt to print APIMethod " + e->name()
            + " for which semantic analysis has not been performed",
            e->location());
    }

    // only print the body if it has contents
    if (e->is_api_method()->body()->statements().size()) {
        text_.increase_indentation();

        // First emit the raw pointer of node_index_ and vec_ci_
        text_.add_line("constexpr size_t simd_width = " +
                       simd_backend::emit_simd_width() +
                       " / (CHAR_BIT*sizeof(value_type));");
        text_.add_line("const size_type* " + emit_rawptr_name("node_index_") +
                       " = node_index_.data();");
        text_.add_line("const size_type* " + emit_rawptr_name("vec_ci_") +
                       " = vec_ci_.data();");
        text_.add_line();

        // create local indexed views
        std::set<std::string> index_decls;
        for (auto const& symbol : e->scope()->locals()) {
            auto var = symbol.second->is_local_variable();
            if (var->is_indexed()) {
                emit_indexed_view(var, index_decls);
                emit_indexed_view_simd(var, index_decls);
                text_.add_line();
            }
        }

        // get loop dimensions
        text_.add_line("int n_ = node_index_.size();");

        // print the vectorized version of the loop
        emit_api_loop(e, "int i_ = 0", "i_ < n_/simd_width", "++i_");
        text_.add_line();

        // delegate the printing of the remainder unvectorized loop
        auto cprinter = cprinter_.get();
        cprinter->clear_text();
        cprinter->set_gutter(text_.get_gutter());
        cprinter->emit_api_loop(e, "int i_ = n_ - n_ % simd_width", "i_ < n_", "++i_");
        text_ << cprinter->text();

        text_.decrease_indentation();
    }

    text_.add_line("}\n");
}

template<targetKind Arch>
void SimdPrinter<Arch>::emit_indexed_view(LocalVariable* var,
                                          std::set<std::string>& decls) {
    auto const& name = var->name();
    auto external = var->external_variable();
    auto const& index_name = external->index_name();
    text_.add_gutter();

    if (decls.find(index_name) == decls.cend()) {
        text_ << "auto ";
        decls.insert(index_name);
    }

    text_ << index_name << " = ";

    if (external->is_cell_indexed_variable()) {
        text_ << "util::indirect_view(util::indirect_view("
              << emit_member_name(index_name) << ", vec_ci_), node_index_);\n";
    }
    else if (external->is_ion()) {
        auto channel = external->ion_channel();
        auto iname = ion_store(channel);
        text_ << "util::indirect_view(" << iname << "." << name << ", "
              << ion_store(channel) << ".index);\n";
    }
    else {
        text_ << " util::indirect_view(" + emit_member_name(index_name) + ", node_index_);\n";
    }
}

template<targetKind Arch>
void SimdPrinter<Arch>::emit_indexed_view_simd(LocalVariable* var,
                                               std::set<std::string>& decls) {
    auto const& name = var->name();
    auto external = var->external_variable();
    auto const& index_name = external->index_name();

    // We need to work with with raw pointers in the vectorized version
    auto channel = var->external_variable()->ion_channel();
    if (channel==ionKind::none) {
        auto raw_index_name = emit_rawptr_name(index_name);
        if (decls.find(raw_index_name) == decls.cend()) {
            text_.add_gutter();
            if (var->is_read())
                text_ << "const ";

            text_ << "value_type* ";
            decls.insert(raw_index_name);
            text_ << raw_index_name << " = "
                  << emit_member_name(index_name) << ".data()";
        }
    }
    else {
        auto iname = ion_store(channel);
        auto ion_var_names = emit_rawptr_ion(iname, name);
        if (decls.find(ion_var_names.first) == decls.cend()) {
            text_.add_gutter();
            text_ << "size_type* ";
            decls.insert(ion_var_names.first);
            text_ << ion_var_names.first << " = " << iname << ".index.data()";
            text_.end_line(";");
        }

        if (decls.find(ion_var_names.second) == decls.cend()) {
            text_.add_gutter();
            if (var->is_read())
                text_ << "const ";

            text_ << "value_type* ";
            decls.insert(ion_var_names.second);
            text_ << ion_var_names.second << " = " << iname << "."
                  << name << ".data()";
        }
    }

    text_.end_line(";");
}

template<targetKind Arch>
void SimdPrinter<Arch>::emit_api_loop(APIMethod* e,
                                      const std::string& start,
                                      const std::string& end,
                                      const std::string& inc) {
    text_.add_gutter();
    text_ << "for (" << start << "; " << end << "; " << inc << ") {";
    text_.end_line();
    text_.increase_indentation();
    text_.add_line("int off_ = i_*simd_width;");

    // First load the index vectors of all involved ions
    std::set<std::string> declared_ion_vars;
    for (auto& symbol : e->scope()->locals()) {
        auto var = symbol.second->is_local_variable();
        if (var->is_indexed()) {
            auto external = var->external_variable();
            auto channel = external->ion_channel();
            std::string cast_type =
                "(const " + simd_backend::emit_index_type() + " *) ";

            std::string vindex_name, index_ptr_name;
            if (channel == ionKind::none) {
                vindex_name = emit_vtmp_name("node_index_");
                index_ptr_name = emit_rawptr_name("node_index_");
            }
            else {
                auto iname = ion_store(channel);
                vindex_name = emit_vindex_name(iname);
                index_ptr_name = emit_rawptr_name(iname);

            }

            if (declared_ion_vars.find(vindex_name) == declared_ion_vars.cend()) {
                declared_ion_vars.insert(vindex_name);
                text_.add_gutter();
                text_ << simd_backend::emit_index_type() << " "
                      << vindex_name << " = ";
                // FIXME: cast should better go inside `emit_load_index()`
                simd_backend::emit_load_index(
                    text_, cast_type + "&" + index_ptr_name + "[off_]");
                text_.end_line(";");
            }

            if (external->is_cell_indexed_variable()) {
                std::string vci_name = emit_vtmp_name("vec_ci_");
                std::string ci_ptr_name = emit_rawptr_name("vec_ci_");

                if (declared_ion_vars.find(vci_name) == declared_ion_vars.cend()) {
                    declared_ion_vars.insert(vci_name);
                    text_.add_gutter();
                    text_ << simd_backend::emit_index_type() << " "
                          << vci_name << " = ";
                    simd_backend::emit_gather_index(text_, "(int *)"+ci_ptr_name, vindex_name, "sizeof(size_type)");
                    text_.end_line(";");
                }
            }
        }
    }

    text_.add_line();
    for (auto& symbol : e->scope()->locals()) {
        auto var = symbol.second->is_local_variable();
        if (is_input(var)) {
            auto ext = var->external_variable();
            text_.add_gutter() << simd_backend::emit_value_type() << " ";
            var->accept(this);
            text_ << " = ";
            ext->accept(this);
            text_.end_line(";");
        }
    }

    text_.add_line();
    e->body()->accept(this);

    std::vector<LocalVariable*> aliased_variables;

    // perform update of external variables (currents etc)
    for (auto &symbol : e->scope()->locals()) {
        auto var = symbol.second->is_local_variable();
        if (is_output(var)      &&
            !is_point_process() &&
            simd_backend::has_scatter()) {
            // We can safely use scatter, but we need to fetch the variable
            // first
            text_.add_line();
            auto ext = var->external_variable();
            auto ext_tmpname = "_" + ext->index_name();
            text_.add_gutter() << simd_backend::emit_value_type() << " "
                               << ext_tmpname << " = ";
            ext->accept(this);
            text_.end_line(";");
            text_.add_gutter();
            text_ << ext_tmpname << " = ";
            simd_backend::emit_binary_op(text_, ext->op(), ext_tmpname,
                                         [this,var](TextBuffer& tb) {
                                             var->accept(this);
                                         });
            text_.end_line(";");
            text_.add_gutter();

            // Build up the index name
            std::string vindex_name, raw_index_name;
            auto channel = ext->ion_channel();
            if (channel != ionKind::none) {
                auto iname = ion_store(channel);
                vindex_name = emit_vindex_name(iname);
                raw_index_name = emit_rawptr_ion(iname, ext->name()).second;
            }
            else {
                vindex_name = emit_vtmp_name("node_index_");
                raw_index_name = emit_rawptr_name(ext->index_name());
            }

            simd_backend::emit_scatter(text_, raw_index_name, vindex_name,
                                       ext_tmpname, "sizeof(value_type)");
            text_.end_line(";");
        }
        else if (is_output(var)) {
            // var is aliased; collect all the aliased variables and we will
            // update them later in a fused loop all at once
            aliased_variables.push_back(var);
        }
    }

    // Emit update code for the aliased variables
    // First, define their scalar equivalents
    constexpr auto scalar_var_prefix = "s_";
    for (auto& v: aliased_variables) {
        text_.add_gutter();
        text_ << "value_type* " << scalar_var_prefix << v->name()
              << " = (value_type*) &" << v->name();
        text_.end_line(";");
    }

    if (aliased_variables.size() > 0) {
        // Update them all in a single loop
        text_.add_line("for (int k_ = 0; k_ < simd_width; ++k_) {");
        text_.increase_indentation();
        for (auto& v: aliased_variables) {
            auto ext = v->external_variable();
            text_.add_gutter();
            text_ << ext->index_name() << "[off_+k_]";
            text_ << (ext->op() == tok::plus ? " += " : " -= ");
            text_ << scalar_var_prefix << v->name() << "[k_]";
            text_.end_line(";");
        }
        text_.decrease_indentation();
        text_.add_line("}");
    }

    text_.decrease_indentation();
    text_.add_line("}");
}

template<targetKind Arch>
void SimdPrinter<Arch>::visit(IndexedVariable *e) {
    std::string vindex_name, value_name;

    auto channel = e->ion_channel();
    if (channel != ionKind::none) {
        auto iname = ion_store(channel);
        vindex_name = emit_vindex_name(iname);
        value_name = emit_rawptr_ion(iname, e->name()).second;
    }
    else {
        vindex_name = emit_vtmp_name("node_index_");
        value_name = emit_rawptr_name(e->index_name());
    }

    simd_backend::emit_gather(text_, value_name, vindex_name, "sizeof(value_type)");
}

template<targetKind Arch>
void SimdPrinter<Arch>::visit(CellIndexedVariable *e) {
    std::string vindex_name, value_name;

    vindex_name = emit_vtmp_name("vec_ci_");
    value_name = emit_rawptr_name(e->index_name());

    simd_backend::emit_gather(text_, vindex_name, value_name, "sizeof(value_type)");
}

template<targetKind Arch>
void SimdPrinter<Arch>::visit(BlockExpression *e) {
    if (!e->is_nested()) {
        std::vector<std::string> names;
        for(auto& symbol : e->scope()->locals()) {
            auto sym = symbol.second.get();
            // input variables are declared earlier, before the
            // block body is printed
            if (is_stack_local(sym) && !is_input(sym)) {
                names.push_back(sym->name());
            }
        }

        if (names.size() > 0) {
            text_.add_gutter() << simd_backend::emit_value_type() << " "
                               << *(names.begin());
            for(auto it=names.begin()+1; it!=names.end(); ++it) {
                text_ << ", " << *it;
            }
            text_.end_line(";");
        }
    }

    for (auto& stmt : e->statements()) {
        if (stmt->is_local_declaration())
            continue;

        // these all must be handled
        text_.add_gutter();
        stmt->accept(this);
        if (not stmt->is_if()) {
            text_.end_line(";");
        }
    }
}

template<targetKind Arch>
void SimdPrinter<Arch>::visit(BinaryExpression *e) {
    auto lhs = e->lhs();
    auto rhs = e->rhs();

    auto emit_lhs = [this, lhs](TextBuffer& tb) {
        lhs->accept(this);
    };
    auto emit_rhs = [this, rhs](TextBuffer& tb) {
        rhs->accept(this);
    };

    try {
        simd_backend::emit_binary_op(text_, e->op(), emit_lhs, emit_rhs);
    } catch (const std::exception& exc) {
        // Rethrow it as a compiler_exception
        throw compiler_exception(
            "SimdPrinter: " + std::string(exc.what()) + ": " +
            yellow(token_string(e->op())), e->location());
    }
}

template<targetKind Arch>
void SimdPrinter<Arch>::visit(AssignmentExpression *e) {
    auto is_memop = [](Expression *e) {
        auto ident = e->is_identifier();
        auto var = (ident) ? ident->symbol()->is_variable() : nullptr;
        return var != nullptr && var->is_range();
    };

    auto lhs = e->lhs();
    auto rhs = e->rhs();
    if (is_memop(lhs)) {
        // that's a store; change printer's state so as not to emit a load
        // instruction for the lhs visit
        simd_backend::emit_store_unaligned(text_,
                                           [this, lhs](TextBuffer&) {
                                               auto range_load_save = range_load_;
                                               range_load_ = false;
                                               lhs->accept(this);
                                               range_load_ = range_load_save;
                                           },
                                           [this, rhs](TextBuffer&) {
                                               rhs->accept(this);
                                           });
    }
    else {
        // that's an ordinary assignment; use base printer
        CPrinter::visit(e);
    }
}


template<targetKind Arch>
void SimdPrinter<Arch>::visit(VariableExpression *e) {
    if (e->is_range() && range_load_) {
        simd_backend::emit_load_unaligned(text_, "&" + e->name() + "[off_]");
    }
    else if (e->is_range()) {
        text_ << "&" << e->name() << "[off_]";
    }
    else {
        simd_backend::emit_set_value(text_, e->name());
    }
}

template<targetKind Arch>
void SimdPrinter<Arch>::visit(UnaryExpression *e) {

    auto arg = e->expression();
    auto emit_arg = [this, arg](TextBuffer& tb) { arg->accept(this); };

    try {
        simd_backend::emit_unary_op(text_, e->op(), emit_arg);
    } catch (std::exception& exc) {
        throw compiler_exception(
            "SimdPrinter: " + std::string(exc.what()) + ": " +
            yellow(token_string(e->op())), e->location());
    }
}

template<targetKind Arch>
void SimdPrinter<Arch>::visit(PowBinaryExpression *e) {
    auto lhs = e->lhs();
    auto rhs = e->rhs();
    auto emit_lhs = [this, lhs](TextBuffer&) { lhs->accept(this); };
    auto emit_rhs = [this, rhs](TextBuffer&) { rhs->accept(this); };
    simd_backend::emit_pow(text_, emit_lhs, emit_rhs);
}

template<targetKind Arch>
void SimdPrinter<Arch>::visit(CallExpression *e) {
    text_ << e->name() << "(off_";
    for (auto& arg: e->args()) {
        text_ << ", ";
        arg->accept(this);
    }
    text_ << ")";
}

template<targetKind Arch>
void SimdPrinter<Arch>::visit(ProcedureExpression *e) {
    auto emit_procedure_unoptimized = [this](ProcedureExpression* e) {
        auto cprinter = cprinter_.get();
        cprinter->clear_text();
        cprinter->set_gutter(text_.get_gutter());
        cprinter->visit(e);
        text_ << cprinter->text();
    };

    if (e->kind() == procedureKind::net_receive) {
        // Use non-vectorized printer for printing net_receive
        emit_procedure_unoptimized(e);
        return;
    }

    // Two versions of each procedure are needed: vectorized and unvectorized
    text_.add_gutter() << "void " << e->name() << "(int off_";
    for(auto& arg : e->args()) {
        text_ << ", " << simd_backend::emit_value_type() << " "
              << arg->is_argument()->name();
    }
    text_ << ") {\n";

    if (!e->scope()) {
        // error: semantic analysis has not been performed
        throw compiler_exception(
            "SimdPrinter attempt to print Procedure " + e->name()
            + " for which semantic analysis has not been performed",
            e->location());
    }

    // print body
    increase_indentation();
    e->body()->accept(this);

    // close the function body
    decrease_indentation();

    text_.add_line("}");
    text_.add_line();

    // Emit also the unvectorised version of the procedure
    emit_procedure_unoptimized(e);
}
