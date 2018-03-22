#include <iostream>
#include <string>

#include "cprinter2.hpp"
#include "expression.hpp"
#include "printerutil.hpp"
#include "io/ostream_wrappers.hpp"
#include "io/prefixbuf.hpp"

using io::indent;
using io::popindent;
using io::quote;

// TODO: some of these will need to be visible for simd printer -- use namespace for qualification?

void emit_procedure_proto(std::ostream&, ProcedureExpression*, const std::string& qualified = "");
void emit_indexed_view(std::ostream&, LocalVariable*);
void emit_state_read(std::ostream&, LocalVariable*);
void emit_state_update(std::ostream&, LocalVariable*);
void emit_api_body(std::ostream&, APIMethod*);

struct cprint {
    Expression* expr_;
    explicit cprint(Expression* expr): expr_(expr) {}

    friend std::ostream& operator<<(std::ostream& out, const cprint& w) {
        CPrinter2 printer(out);
        return w.expr_->accept(&printer), out;
    }
};

void confirm_ok(Expression* expr, const std::string context) {
    return
        !expr? throw compiler_exception("missing expression for "+context):
        !expr->scope()? throw compiler_exception("printer invoked before semantic pass for "+context):
        void();
}

static std::string ion_state_field(std::string ion_name) {
    return "ion_"+ion_name+"_";
}

static std::string ion_state_index(std::string ion_name) {
    return "ion_"+ion_name+"_index_";
}

std::string emit_cpp_source(const Module& module_, const std::string& ns) {
    const char* arb_header_prefix = "";
    std::string name = module_.module_name();
    std::string class_name = "mechanism_cpu_"+name;
    auto ns_components = namespace_components(ns);

    NetReceiveExpression* net_receive = find_net_receive(module_);
    APIMethod* init_api = find_api_method(module_, "nrn_init");
    APIMethod* state_api = find_api_method(module_, "nrn_state");
    APIMethod* current_api = find_api_method(module_, "nrn_current");
    APIMethod* write_ions_api = find_api_method(module_, "write_ions");

    // init_api, state_api, current_api methods are mandatory:

    confirm_ok(init_api, "nrn_init");
    confirm_ok(state_api, "nrn_state");
    confirm_ok(current_api, "nrn_current");

    auto vars = local_module_variables(module_);
    auto ion_deps = module_.ion_deps();
    std::string fingerprint = "<placeholder>";

    io::pfxstringstream out;

    out <<
        "#include <cmath>\n"
        "#include <cstddef>\n"
        "#include <memory>\n"
        "#include <" << arb_header_prefix << "backends/multicore/fvm.hpp>\n"
        "#include <" << arb_header_prefix << "math.hpp>\n"
        "#include <" << arb_header_prefix << "mechanism.hpp>\n"
        "\n"
        << namespace_declaration_open(ns_components) <<
        "\n"
        "using backend = ::arb::multicore::backend;\n"
        "using base = ::arb::multicore::mechanism;\n"
        "using value_type = base::value_type;\n"
        "using ::arb::util::indirect_view;\n"
        "using ::arb::math::min;\n"
        "using ::arb::math::max;\n"
        "using ::arb::math::exprelr;\n"
        "\n"
        "class " << class_name << ": public base {\n"
        "public:\n" << indent <<
        "const mechanism_fingerprint& fingerprint() const override {\n" << indent <<
        "static mechanism_fingerprint hash = " << quote(fingerprint) << ";\n"
        "return hash;\n" << popindent <<
        "}\n"
        "std::string internal_name() const override { return " << quote(name) << "; }\n"
        "mechanismKind kind() const override { return " << module_kind_str(module_) << "; }\n"
        "mechanism_ptr clone() const override { return mechanism_ptr(new " << class_name << "()); }\n"
        "\n"
        "void nrn_init() override;\n"
        "void nrn_state() override;\n"
        "void nrn_current() override;\n"
        "void write_ions() override;\n";

    net_receive && out <<
        "void deliver_events(deliverable_event_stream::state events) override;\n"
        "void net_receive(int i_, value_type weight);\n";

    out <<
        "\n" << popindent <<
        "protected:\n" << indent <<
        "using ionKind = ::arb::ionKind;\n\n"
        "std::size_t object_sizeof() const override { return sizeof(*this); }\n";

    io::separator sep("\n", ",\n");
    if (!vars.scalars.empty()) {
        out <<
            "mechanism_global_table global_table() {\n" << indent <<
            "return {" << indent;

        for (const auto& scalar: vars.scalars) {
            auto memb = scalar->name();
            out << sep << "{" << quote(memb) << ", &" << memb << "}";
        }
        out << popindent << "\n};\n" << popindent << "}\n";
    }

    if (!vars.arrays.empty()) {
        out <<
            "mechanism_field_table field_table() {\n" << indent <<
            "return {" << indent;

        sep.reset();
        for (const auto& array: vars.arrays) {
            auto memb = array->name();
            auto dflt = array->value();
            out << sep << "{" << quote(memb) << ", &" << memb << ", " << as_c_double(dflt) << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";
    }

    if (!ion_deps.empty()) {
        out <<
            "mechanism_ion_state_table ion_state_table() {\n" << indent <<
            "return {" << indent;

        sep.reset();
        for (const auto& dep: ion_deps) {
            out << sep << "{ionKind::" << dep.name << ", &" << ion_state_field(dep.name) << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";

        sep.reset();
        out << "mechanism_ion_index_table ion_index_table() {\n" << indent << "return {" << indent;
        for (const auto& dep: ion_deps) {
            out << sep << "{ionKind::" << dep.name << ", &" << ion_state_index(dep.name) << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";
    }

    out << popindent << "\n"
        "private:\n" << indent;

    for (const auto& scalar: vars.scalars) {
        out << "value_type " << scalar->name() <<  " = " << as_c_double(scalar->value()) << ";\n";
    }
    for (const auto& array: vars.arrays) {
        out << "view " << array->name() << ";\n";
    }
    for (const auto& dep: ion_deps) {
        out << "ion_state " << ion_state_field(dep.name) << ";\n";
        out << "iarray " << ion_state_index(dep.name) << ";\n";
    }

    for (auto proc: normal_procedures(module_)) {
        emit_procedure_proto(out, proc);
        out << ";\n";
    }

    out << popindent <<
        "};\n\n"
        "template <typename B> ::arb::concrete_mech_ptr<B> make_mechanism_" <<name << "();\n"
        "template <> ::arb::concrete_mech_ptr<backend> make_mechanism_" << name << "<backend>() {\n" << indent <<
        "return ::arb::concrete_mech_ptr<backend>(new " << class_name << "());\n" << popindent <<
        "}\n\n";

    // Nrn methods:

    net_receive && out <<
        "void " << class_name << "::deliver_events(deliverable_event_stream::state events) {\n" << indent <<
        "auto ncell = events.n_streams();\n"
        "for (size_type c = 0; c<ncell; ++c) {\n" << indent <<
        "auto begin = events.begin_marked(c);\n"
        "auto end = events.end_marked(c);\n"
        "for (auto p = begin; p<end; ++p) {\n" << indent <<
        "if (p->mech_id==mechanism_id_) net_receive(p->mech_index, p->weight);\n" << popindent <<
        "}\n" << popindent <<
        "}\n" << popindent <<
        "}\n"
        "\n"
        "void " << class_name << "::net_receive(int i_, value_type weight) {\n" << indent <<
        cprint(net_receive->body()) << popindent <<
        "}\n\n";

    out << "void " << class_name << "::nrn_init() {\n" << indent;
    emit_api_body(out, init_api);
    out << popindent << "}\n\n";

    out << "void " << class_name << "::nrn_state() {\n" << indent;
    emit_api_body(out, state_api);
    out << popindent << "}\n\n";

    out << "void " << class_name << "::nrn_current() {\n" << indent;
    emit_api_body(out, current_api);
    out << popindent << "}\n\n";

    out << "void " << class_name << "::write_ions() {\n" << indent;
    emit_api_body(out, write_ions_api);
    out << popindent << "}\n\n";

    // Mechanism procedures

    for (auto proc: normal_procedures(module_)) {
        emit_procedure_proto(out, proc, class_name);
        out <<
            " {" << indent <<
            cprint(proc->body()) << popindent <<
            "}\n\n";
    }

    out << namespace_declaration_close(ns_components);
    return out.str();
}

void emit_procedure_proto(std::ostream& out, ProcedureExpression* e, const std::string& qualified) {
    out << "void " << qualified << (qualified.empty()? "": "::") << e->name() << "(int i_";
    for (auto& arg: e->args()) {
        out << ", value_type " << arg->is_argument()->name();
    }
    out << ")";
}

void emit_indexed_view(std::ostream& out, IndexedVariable* external) {
    std::string data_var, ion_pfx;
    std::string index = "node_index_";

    if (external->is_ion()) {
        ion_pfx = "ion_"+to_string(external->ion_channel())+"_";
        index = ion_pfx+"index_";
    }

    switch (external->data_source()) {
    case sourceKind::voltage:
        data_var="vec_v_";
        break;
    case sourceKind::current:
        data_var="vec_i_";
        break;
    case sourceKind::dt:
        data_var="vec_dt_";
        break;
    case sourceKind::ion_current:
        data_var=ion_pfx+".current_density";
        break;
    case sourceKind::ion_revpot:
        data_var=ion_pfx+".reversal_potential";
        break;
    case sourceKind::ion_iconc:
        data_var=ion_pfx+".internal_concentration";
        break;
    case sourceKind::ion_econc:
        data_var=ion_pfx+".external_concentration";
        break;
    default:
        throw compiler_exception("unrecognized indexed data source", external->location());
    }

    auto view_var = external->index_name();
    out << "auto " << view_var << " = indirect_view(" << data_var << ", " << index << ");\n";
}

void emit_state_read(std::ostream& out, LocalVariable* local) {
    out << "value_type " << cprint(local) << " = ";

    if (local->is_read()) {
        out << cprint(local->external_variable()) << ";\n";
    }
    else {
        out << "0;\n";
    }
}

void emit_state_update(std::ostream& out, Symbol* from, IndexedVariable* external) {
    if (external->is_write()) {
        const char* op = external->op()==tok::plus? "+=": "-=";
        out << cprint(external) << op << cprint(from) << ";\n";
    }
}

void emit_api_body(std::ostream& out, APIMethod* method) {
    auto body = method->body();
    auto indexed_vars = indexed_locals(method->scope());

    if (!body->statements().empty()) {
        for (auto& sym: indexed_vars) {
            emit_indexed_view(out, sym->external_variable());
        }
        out <<
            "int n_ = node_index_.size();\n"
            "for (int i_ = 0; i_ < n_; ++i_) {\n" << indent;

        for (auto& sym: indexed_vars) {
            emit_state_read(out, sym);
        }
        out << cprint(body);

        for (auto& sym: indexed_vars) {
            emit_state_update(out, sym, sym->external_variable());
        }
        out << popindent << "}\n";
    }
}

// CPrinter methods:

void CPrinter2::visit(IdentifierExpression *e) {
    e->symbol()->accept(this);
}

void CPrinter2::visit(LocalVariable* sym) {
    out_ << sym->name();
}

void CPrinter2::visit(VariableExpression *sym) {
    out_ << sym->name() << (sym->is_range()? "[i_]": "");
}

void CPrinter2::visit(IndexedVariable *sym) {
    auto view_var = sym->index_name();
    out_ << view_var << "[i_]";
}

void CPrinter2::visit(CallExpression* e) {
    out_ << e->name() << "(i_";
    for (auto& arg: e->args()) {
        out_ << ", ";
        arg->accept(this);
    }
    out_ << ")";
}

void CPrinter2::visit(BlockExpression* block) {
    // Only include local declarations in outer-most block.
    if (!block->is_nested()) {
        auto locals = pure_locals(block->scope());
        if (!locals.empty()) {
            out_ << "value_type ";
            io::separator sep(", ");
            for (auto local: locals) {
                out_ << sep << local->name();
            }
            out_ << ";\n";
        }
    }

    for (auto& stmt: block->statements()) {
        if (!stmt->is_local_declaration()) {
            stmt->accept(this);
            out_ << (stmt->is_if()? "": ";\n");
        }
    }
}

