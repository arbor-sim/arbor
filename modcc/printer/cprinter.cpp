#include <cmath>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_set>

#include "expression.hpp"
#include "io/ostream_wrappers.hpp"
#include "io/prefixbuf.hpp"
#include "printer/cexpr_emit.hpp"
#include "printer/cprinter.hpp"
#include "printer/printeropt.hpp"
#include "printer/printerutil.hpp"

using io::indent;
using io::popindent;
using io::quote;

constexpr bool with_profiling() {
#ifdef ARB_HAVE_PROFILING
    return true;
#else
    return false;
#endif
}

struct index_prop {
    std::string source_var; // array holding the indices
    std::string index_name; // index into the array
    bool        node_index; // node index (cv) or cell index
    bool operator==(const index_prop& other) const {
        return (source_var == other.source_var) && (index_name == other.index_name);
    }
};

void emit_procedure_proto(std::ostream&, ProcedureExpression*, const std::string& qualified = "");
void emit_simd_procedure_proto(std::ostream&, ProcedureExpression*, const std::string& qualified = "");
void emit_masked_simd_procedure_proto(std::ostream&, ProcedureExpression*, const std::string& qualified = "");

void emit_api_body(std::ostream&, APIMethod*);
void emit_simd_api_body(std::ostream&, APIMethod*, const std::vector<VariableExpression*>& scalars);

void emit_simd_index_initialize(std::ostream& out, const std::list<index_prop>& indices, simd_expr_constraint constraint);

void emit_simd_body_for_loop(std::ostream& out,
                             BlockExpression* body,
                             const std::vector<LocalVariable*>& indexed_vars,
                             const std::list<index_prop>& indices,
                             const simd_expr_constraint& constraint);

void emit_simd_for_loop_per_constraint(std::ostream& out, BlockExpression* body,
                                       const std::vector<LocalVariable*>& indexed_vars,
                                       const std::list<index_prop>& indices,
                                       const simd_expr_constraint& constraint,
                                       std::string constraint_name);

struct cprint {
    Expression* expr_;
    explicit cprint(Expression* expr): expr_(expr) {}

    friend std::ostream& operator<<(std::ostream& out, const cprint& w) {
        CPrinter printer(out);
        return w.expr_->accept(&printer), out;
    }
};

struct simdprint {
    Expression* expr_;
    bool is_indirect_ = false;
    bool is_masked_ = false;
    std::unordered_set<std::string> scalars_;

    explicit simdprint(Expression* expr, const std::vector<VariableExpression*>& scalars): expr_(expr) {
        for (const auto& s: scalars) {
            scalars_.insert(s->name());
        }
    }

    void set_indirect_index() {
        is_indirect_ = true;
    }
    void set_masked() {
        is_masked_ = true;
    }

    friend std::ostream& operator<<(std::ostream& out, const simdprint& w) {
        SimdPrinter printer(out);
        if(w.is_masked_) {
            printer.set_input_mask("mask_input_");
        }
        printer.set_var_indexed(w.is_indirect_);
        printer.save_scalar_names(w.scalars_);
        return w.expr_->accept(&printer), out;
    }
};

static std::string ion_state_field(std::string ion_name) {
    return "ion_"+ion_name+"_";
}

static std::string ion_state_index(std::string ion_name) {
    return "ion_"+ion_name+"_index_";
}

std::string emit_cpp_source(const Module& module_, const printer_options& opt) {
    std::string name = module_.module_name();
    std::string class_name = "mechanism_cpu_"+name;
    auto ns_components = namespace_components(opt.cpp_namespace);

    NetReceiveExpression* net_receive = find_net_receive(module_);
    APIMethod* init_api = find_api_method(module_, "nrn_init");
    APIMethod* state_api = find_api_method(module_, "nrn_state");
    APIMethod* current_api = find_api_method(module_, "nrn_current");
    APIMethod* write_ions_api = find_api_method(module_, "write_ions");

    bool with_simd = opt.simd.abi!=simd_spec::none;

    // init_api, state_api, current_api methods are mandatory:

    assert_has_scope(init_api, "nrn_init");
    assert_has_scope(state_api, "nrn_state");
    assert_has_scope(current_api, "nrn_current");

    auto vars = local_module_variables(module_);
    auto ion_deps = module_.ion_deps();
    std::string fingerprint = "<placeholder>";

    auto profiler_enter = [name, opt](const char* region_prefix) -> std::string {
        static std::regex invalid_profile_chars("[^a-zA-Z0-9]");

        if (opt.profile) {
            std::string region_name = region_prefix;
            region_name += '_';
            region_name += std::regex_replace(name, invalid_profile_chars, "");

            return
                "{\n"
                "    static auto id = ::arb::profile::profiler_region_id(\""
                + region_name + "\");\n"
                "    ::arb::profile::profiler_enter(id);\n"
                "}\n";
        }
        else return "";
    };

    auto profiler_leave = [opt]() -> std::string {
        return opt.profile? "::arb::profile::profiler_leave();\n": "";
    };

    io::pfxstringstream out;

    out <<
        "#include <algorithm>\n"
        "#include <cmath>\n"
        "#include <cstddef>\n"
        "#include <memory>\n"
        "#include <" << arb_private_header_prefix() << "backends/multicore/mechanism.hpp>\n"
        "#include <" << arb_header_prefix() << "math.hpp>\n";

    opt.profile &&
        out << "#include <" << arb_header_prefix() << "profile/profiler.hpp>\n";

    if (with_simd) {
        out << "#include <" << arb_header_prefix() << "simd/simd.hpp>\n";
        out << "#undef NDEBUG\n";
        out << "#include <cassert>\n";
    }

    out <<
        "\n" << namespace_declaration_open(ns_components) <<
        "\n"
        "using backend = ::arb::multicore::backend;\n"
        "using base = ::arb::multicore::mechanism;\n"
        "using value_type = base::value_type;\n"
        "using size_type = base::size_type;\n"
        "using index_type = base::index_type;\n"
        "using ::arb::math::exprelr;\n"
        "using ::arb::math::safeinv;\n"
        "using ::std::abs;\n"
        "using ::std::cos;\n"
        "using ::std::exp;\n"
        "using ::std::log;\n"
        "using ::std::max;\n"
        "using ::std::min;\n"
        "using ::std::pow;\n"
        "using ::std::sin;\n"
        "\n";

    if (with_simd) {
        out <<
            "namespace S = ::arb::simd;\n"
            "using S::index_constraint;\n"
            "using S::simd_cast;\n"
            "using S::indirect;\n"
            "using S::assign;\n";

        out << "static constexpr unsigned vector_length_ = ";
        if (opt.simd.size == no_size) {
            out << "S::simd_abi::native_width<::arb::fvm_value_type>::value;\n";
        } else {
            out << opt.simd.size << ";\n";
        }

        out << "static constexpr unsigned simd_width_ = ";
        if (opt.simd.width == no_size) {
            out << " vector_length_ ? vector_length_ : " << opt.simd.default_width << ";\n";
        } else {
            out << opt.simd.width << ";\n";
        }

        std::string abi = "S::simd_abi::";
        switch (opt.simd.abi) {
        case simd_spec::avx:    abi += "avx";    break;
        case simd_spec::avx2:   abi += "avx2";   break;
        case simd_spec::avx512: abi += "avx512"; break;
        case simd_spec::neon:   abi += "neon";   break;
        case simd_spec::sve:    abi += "sve";    break;
        case simd_spec::native: abi += "native"; break;
        default:
            abi += "default_abi"; break;
        }

        out <<
            "using simd_value = S::simd<::arb::fvm_value_type, vector_length_, " << abi << ">;\n"
            "using simd_index = S::simd<::arb::fvm_index_type, vector_length_, " << abi << ">;\n"
            "using simd_mask  = S::simd_mask<::arb::fvm_value_type, vector_length_, "<< abi << ">;\n"
            "\n"
            "inline simd_value safeinv(simd_value x) {\n"
            "    simd_value ones = simd_cast<simd_value>(1.0);\n"
            "    auto mask = S::cmp_eq(S::add(x,ones), ones);\n"
            "    S::where(mask, x) = simd_cast<simd_value>(DBL_EPSILON);\n"
            "    return S::div(ones, x);\n"
            "}\n"
            "\n";
    }

    out <<
        "class " << class_name << ": public base {\n"
        "public:\n" << indent <<
        "const ::arb::mechanism_fingerprint& fingerprint() const override {\n" << indent <<
        "static ::arb::mechanism_fingerprint hash = " << quote(fingerprint) << ";\n"
        "return hash;\n" << popindent <<
        "}\n"
        "std::string internal_name() const override { return " << quote(name) << "; }\n"
        "::arb::mechanismKind kind() const override { return " << module_kind_str(module_) << "; }\n"
        "::arb::mechanism_ptr clone() const override { return ::arb::mechanism_ptr(new " << class_name << "()); }\n"
        "\n"
        "void nrn_init() override;\n"
        "void nrn_state() override;\n"
        "void nrn_current() override;\n"
        "void write_ions() override;\n";

    net_receive && out <<
        "void deliver_events(deliverable_event_stream::state events) override;\n"
        "void net_receive(int i_, value_type weight);\n";

    with_simd && out << "unsigned simd_width() const override { return simd_width_; }\n";

    out <<
        "\n" << popindent <<
        "protected:\n" << indent <<
        "std::size_t object_sizeof() const override { return sizeof(*this); }\n";

    io::separator sep("\n", ",\n");
    if (!vars.scalars.empty()) {
        out <<
            "mechanism_global_table global_table() override {\n" << indent <<
            "return {" << indent;

        for (const auto& scalar: vars.scalars) {
            auto memb = scalar->name();
            out << sep << "{" << quote(memb) << ", &" << memb << "}";
        }
        out << popindent << "\n};\n" << popindent << "}\n";
    }

    if (!vars.arrays.empty()) {
        out <<
            "mechanism_field_table field_table() override {\n" << indent <<
            "return {" << indent;

        sep.reset();
        for (const auto& array: vars.arrays) {
            auto memb = array->name();
            out << sep << "{" << quote(memb) << ", &" << memb << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";

        out <<
            "mechanism_field_default_table field_default_table() override {\n" << indent <<
            "return {" << indent;

        sep.reset();
        for (const auto& array: vars.arrays) {
            auto memb = array->name();
            auto dflt = array->value();
            if (!std::isnan(dflt)) {
                out << sep << "{" << quote(memb) << ", " << as_c_double(dflt) << "}";
            }
        }
        out << popindent << "\n};" << popindent << "\n}\n";

        out <<
            "mechanism_state_table state_table() override {\n" << indent <<
            "return {" << indent;

        sep.reset();
        for (const auto& array: vars.arrays) {
            auto memb = array->name();
            if(array->is_state()) {
                out << sep << "{" << quote(memb) << ", &" << memb << "}";
            }
        }
        out << popindent << "\n};" << popindent << "\n}\n";

    }

    if (!ion_deps.empty()) {
        out <<
            "mechanism_ion_state_table ion_state_table() override {\n" << indent <<
            "return {" << indent;

        sep.reset();
        for (const auto& dep: ion_deps) {
            out << sep << "{\"" << dep.name << "\", &" << ion_state_field(dep.name) << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";

        sep.reset();
        out << "mechanism_ion_index_table ion_index_table() override {\n" << indent << "return {" << indent;
        for (const auto& dep: ion_deps) {
            out << sep << "{\"" << dep.name << "\", &" << ion_state_index(dep.name) << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";
    }

    out << popindent << "\n"
        "private:\n" << indent;

    for (const auto& scalar: vars.scalars) {
        out << "value_type " << scalar->name() <<  " = " << as_c_double(scalar->value()) << ";\n";
    }
    for (const auto& array: vars.arrays) {
        out << "value_type* " << array->name() << ";\n";
    }
    for (const auto& dep: ion_deps) {
        out << "ion_state_view " << ion_state_field(dep.name) << ";\n";
        out << "iarray " << ion_state_index(dep.name) << ";\n";
    }

    for (auto proc: normal_procedures(module_)) {
        if (with_simd) {
            emit_simd_procedure_proto(out, proc);
            out << ";\n";
            emit_masked_simd_procedure_proto(out, proc);
            out << ";\n";
        } else {
            emit_procedure_proto(out, proc);
            out << ";\n";
        }
    }

    out << popindent <<
        "};\n\n"
        "template <typename B> ::arb::concrete_mech_ptr<B> make_mechanism_" <<name << "();\n"
        "template <> ::arb::concrete_mech_ptr<backend> make_mechanism_" << name << "<backend>() {\n" << indent <<
        "return ::arb::concrete_mech_ptr<backend>(new " << class_name << "());\n" << popindent <<
        "}\n\n";

    // Nrn methods:

    if (net_receive) {
        const std::string weight_arg = net_receive->args().empty() ? "weight" : net_receive->args().front()->is_argument()->name();
        out <<
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
            "void " << class_name << "::net_receive(int i_, value_type " << weight_arg << ") {\n" << indent <<
            cprint(net_receive->body()) << popindent <<
            "}\n\n";
    }

    auto emit_body = [&](APIMethod *p) {
        if (with_simd) {
            emit_simd_api_body(out, p, vars.scalars);
        }
        else {
            emit_api_body(out, p);
        }
    };

    out << "void " << class_name << "::nrn_init() {\n" << indent;
    emit_body(init_api);
    out << popindent << "}\n\n";

    out << "void " << class_name << "::nrn_state() {\n" << indent;
    out << profiler_enter("advance_integrate_state");
    emit_body(state_api);
    out << profiler_leave();
    out << popindent << "}\n\n";

    out << "void " << class_name << "::nrn_current() {\n" << indent;
    out << profiler_enter("advance_integrate_current");
    emit_body(current_api);
    out << profiler_leave();
    out << popindent << "}\n\n";

    out << "void " << class_name << "::write_ions() {\n" << indent;
    emit_body(write_ions_api);
    out << popindent << "}\n\n";

    // Mechanism procedures

    for (auto proc: normal_procedures(module_)) {
        if (with_simd) {
            emit_simd_procedure_proto(out, proc, class_name);
            auto simd_print = simdprint(proc->body(), vars.scalars);
            out << " {\n" << indent << simd_print << popindent <<  "}\n\n";

            emit_masked_simd_procedure_proto(out, proc, class_name);
            auto masked_print = simdprint(proc->body(), vars.scalars);
            masked_print.set_masked();
            out << " {\n" << indent << masked_print << popindent << "}\n\n";
        } else {
            emit_procedure_proto(out, proc, class_name);
            out <<
                " {\n" << indent <<
                cprint(proc->body()) << popindent <<
                "}\n\n";
        }
    }

    out << namespace_declaration_close(ns_components);
    return out.str();
}

// Scalar printing:

void CPrinter::visit(IdentifierExpression *e) {
    e->symbol()->accept(this);
}

void CPrinter::visit(LocalVariable* sym) {
    out_ << sym->name();
}

void CPrinter::visit(VariableExpression *sym) {
    out_ << sym->name() << (sym->is_range()? "[i_]": "");
}

void CPrinter::visit(CallExpression* e) {
    out_ << e->name() << "(i_";
    for (auto& arg: e->args()) {
        out_ << ", ";
        arg->accept(this);
    }
    out_ << ")";
}

void CPrinter::visit(BlockExpression* block) {
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

static std::string index_i_name(const std::string& index_var) {
    return index_var+"i_";
}

void emit_procedure_proto(std::ostream& out, ProcedureExpression* e, const std::string& qualified) {
    out << "void " << qualified << (qualified.empty()? "": "::") << e->name() << "(int i_";
    for (auto& arg: e->args()) {
        out << ", value_type " << arg->is_argument()->name();
    }
    out << ")";
}

namespace {
    // Convenience I/O wrapper for emitting indexed access to an external variable.

    struct deref {
        indexed_variable_info d;
        deref(indexed_variable_info d): d(d) {}

        friend std::ostream& operator<<(std::ostream& o, const deref& wrap) {
            auto index_var = wrap.d.cell_index_var.empty() ? wrap.d.node_index_var : wrap.d.cell_index_var;
            return o << wrap.d.data_var << '['
                     << (wrap.d.scalar()? "0": index_i_name(index_var)) << ']';
        }
    };
}

void emit_state_read(std::ostream& out, LocalVariable* local) {
    out << "value_type " << cprint(local) << " = ";

    if (local->is_read()) {
        auto d = decode_indexed_variable(local->external_variable());
        if (d.scale != 1) {
            out << as_c_double(d.scale) << "*";
        }
        out << deref(d) << ";\n";
    }
    else {
        out << "0;\n";
    }
}

void emit_state_update(std::ostream& out, Symbol* from, IndexedVariable* external) {
    if (!external->is_write()) return;

    auto d = decode_indexed_variable(external);
    double coeff = 1./d.scale;

    if (d.readonly) {
        throw compiler_exception("Cannot assign to read-only external state: "+external->to_string());
    }

    if (d.accumulate) {
        out << deref(d) << " = fma(";
        if (coeff != 1) {
            out << as_c_double(coeff) << '*';
        }
        out << "weight_[i_], " << from->name() << ", " << deref(d) << ");\n";
    }
    else {
        out << deref(d) << " = ";
        if (coeff != 1) {
            out << as_c_double(coeff) << '*';
        }
        out << from->name() << ";\n";
    }
}

void emit_api_body(std::ostream& out, APIMethod* method) {
    auto body = method->body();
    auto indexed_vars = indexed_locals(method->scope());

    std::list<index_prop> indices;
    for (auto& sym: indexed_vars) {
        auto d = decode_indexed_variable(sym->external_variable());
        if (!d.scalar()) {
            index_prop node_idx = {d.node_index_var, "i_", true};
            auto it = std::find(indices.begin(), indices.end(), node_idx);
            if (it == indices.end()) indices.push_front(node_idx);
            if (!d.cell_index_var.empty()) {
                index_prop cell_idx = {d.cell_index_var, index_i_name(d.node_index_var), false};
                auto it = std::find(indices.begin(), indices.end(), cell_idx);
                if (it == indices.end()) indices.push_back(cell_idx);
            }
        }
    }

    if (!body->statements().empty()) {
        out <<
            "int n_ = width_;\n"
            "for (int i_ = 0; i_ < n_; ++i_) {\n" << indent;

        for (auto index: indices) {
            out << "auto " << index_i_name(index.source_var) << " = " << index.source_var << "[" << index.index_name << "];\n";
        }

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

// SIMD printing:

void SimdPrinter::visit(IdentifierExpression *e) {
    e->symbol()->accept(this);
}

void SimdPrinter::visit(LocalVariable* sym) {
    out_ << sym->name();
}

void SimdPrinter::visit(VariableExpression *sym) {
    if (sym->is_range()) {
        auto index = is_indirect_? "index_": "i_";
        out_ << "simd_cast<simd_value>(indirect(" << sym->name() << "+" << index << ", simd_width_))";
    }
    else {
        out_ << sym->name();
    }
}

void SimdPrinter::visit(AssignmentExpression* e) {
    if (!e->lhs() || !e->lhs()->is_identifier() || !e->lhs()->is_identifier()->symbol()) {
        throw compiler_exception("Expect symbol on lhs of assignment: "+e->to_string());
    }

    Symbol* lhs = e->lhs()->is_identifier()->symbol();
 
    bool cast = false;
    if (auto id = e->rhs()->is_identifier()) {
        if (scalars_.count(id->name())) cast = true;
    }
    if (e->rhs()->is_number()) cast = true;
    if (scalars_.count(e->lhs()->is_identifier()->name()))  cast = false;

    if (lhs->is_variable() && lhs->is_variable()->is_range()) {
        if(is_indirect_)
            out_ << "indirect(" << lhs->name() << "+index_, simd_width_) = ";
        else
            out_ << "indirect(" << lhs->name() << "+i_, simd_width_) = ";

        if (!input_mask_.empty())
            out_ << "S::where(" << input_mask_ << ", ";

        if (cast) out_ << "simd_cast<simd_value>(";
        e->rhs()->accept(this);
        if (cast) out_ << ")";

        if (!input_mask_.empty())
            out_ << ")";
    }
    else {
        out_ << "assign(" << lhs->name() << ", ";
        if (auto rhs = e->rhs()->is_identifier()) {
            if (auto sym = rhs->symbol()) {
                // We shouldn't call the rhs visitor in this case because it automatically casts indirect expressions
                if (sym->is_variable() && sym->is_variable()->is_range()) {
                    auto index = is_indirect_ ? "index_" : "i_";
                    out_ << "indirect(" << rhs->name() << "+" << index << ", simd_width_))";
                    return;
                }
            }
        }
        e->rhs()->accept(this);
        out_ << ")";
    }
}

void SimdPrinter::visit(CallExpression* e) {
    if(is_indirect_)
        out_ << e->name() << "(index_";
    else
        out_ << e->name() << "(i_";
    for (auto& arg: e->args()) {
        out_ << ", ";
        arg->accept(this);
    }
    out_ << ")";
}

void SimdPrinter::visit(BlockExpression* block) {
    // Only include local declarations in outer-most block.
    if (!block->is_nested()) {
        auto locals = pure_locals(block->scope());
        if (!locals.empty()) {
            out_ << "simd_value ";
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
            if (!stmt->is_if() && !stmt->is_block()) {
                out_ << ";\n";
            }
        }
    }
}

void emit_simd_procedure_proto(std::ostream& out, ProcedureExpression* e, const std::string& qualified) {
    out << "void " << qualified << (qualified.empty()? "": "::") << e->name() << "(index_type i_";
    for (auto& arg: e->args()) {
        out << ", const simd_value& " << arg->is_argument()->name();
    }
    out << ")";
}

void emit_masked_simd_procedure_proto(std::ostream& out, ProcedureExpression* e, const std::string& qualified) {
    out << "void " << qualified << (qualified.empty()? "": "::") << e->name()
    << "(index_type i_, simd_mask mask_input_";
    for (auto& arg: e->args()) {
        out << ", const simd_value& " << arg->is_argument()->name();
    }
    out << ")";
}

void emit_simd_state_read(std::ostream& out, LocalVariable* local, simd_expr_constraint constraint) {
    out << "simd_value " << local->name();

    if (local->is_read()) {
        auto d = decode_indexed_variable(local->external_variable());
        if (d.scalar()) {
            out << " = simd_cast<simd_value>(" << d.data_var
                << "[0]);\n";
        }
        else {
            if (d.cell_index_var.empty()) {
                switch (constraint) {
                    case simd_expr_constraint::contiguous:
                        out << ";\n"
                            << "assign(" << local->name() << ", indirect(" << d.data_var
                            << " + " << index_i_name(d.node_index_var) << ", simd_width_));\n";
                        break;
                    case simd_expr_constraint::constant:
                        out << " = simd_cast<simd_value>(" << d.data_var
                            << "[" << index_i_name(d.node_index_var)  << "]);\n";
                        break;
                    default:
                        out << ";\n"
                            << "assign(" << local->name() << ", indirect(" << d.data_var
                            << ", " << index_i_name(d.node_index_var) << ", simd_width_, constraint_category_));\n";
                }
            }
            else {
                out << ";\n"
                    << "assign(" << local->name() << ", indirect(" << d.data_var
                    << ", " << index_i_name(d.cell_index_var) << ", simd_width_, index_constraint::none));\n";
            }
        }

        if (d.scale != 1) {
            out << local->name() << " = S::mul(" << local->name() << ", simd_cast<simd_value>(" << d.scale << "));\n";
        }
    }
    else {
        out << " = simd_cast<simd_value>(0);\n";
    }
}

void emit_simd_state_update(std::ostream& out, Symbol* from, IndexedVariable* external, simd_expr_constraint constraint) {
    if (!external->is_write()) return;

    auto d = decode_indexed_variable(external);;
    double coeff = 1./d.scale;

    if (d.readonly) {
        throw compiler_exception("Cannot assign to read-only external state: "+external->to_string());
    }

    if (d.accumulate) {
        if (d.cell_index_var.empty()) {
            switch (constraint) {
                case simd_expr_constraint::contiguous:
                {
                    std::string tempvar = "t_" + external->name();
                    out << "simd_value " << tempvar << ";\n"
                        << "assign(" << tempvar << ", indirect(" << d.data_var << " + " << index_i_name(d.node_index_var) << ", simd_width_));\n";
                    if (coeff != 1) {
                        out << tempvar << " = S::fma(S::mul(w_, simd_cast<simd_value>(" << as_c_double(coeff) << "))," << from->name() << ", " << tempvar << ");\n";
                    } else {
                        out << tempvar << " = S::fma(w_, " << from->name() << ", " << tempvar << ");\n";
                    }
                    out << "indirect(" << d.data_var << " + " << index_i_name(d.node_index_var) << ", simd_width_) = " << tempvar << ";\n";
                    break;
                }
                case simd_expr_constraint::constant:
                {
                    out << "indirect(" << d.data_var << ", simd_cast<simd_index>(" << index_i_name(d.node_index_var) << "), simd_width_, constraint_category_)";
                    if (coeff != 1) {
                        out << " += S::mul(w_, S::mul(simd_cast<simd_value>(" << as_c_double(coeff) << "), " << from->name() << "));\n";
                    } else {
                        out << " += S::mul(w_, " << from->name() << ");\n";
                    }
                    break;
                }
                default :
                {
                    out << "indirect(" << d.data_var << ", " << index_i_name(d.node_index_var) << ", simd_width_, constraint_category_)";
                    if (coeff != 1) {
                        out << " += S::mul(w_, S::mul(simd_cast<simd_value>(" << as_c_double(coeff) << "), " << from->name() << "));\n";
                    } else {
                        out << " += S::mul(w_, " << from->name() << ");\n";
                    }
                }
            }
        } else {
            out << "indirect(" << d.data_var << ", " << index_i_name(d.cell_index_var) << ", simd_width_, index_constraint::none)";
            if (coeff != 1) {
                out << " += S::mul(w_, S::mul(simd_cast<simd_value>(" << as_c_double(coeff) << "), " << from->name() << "));\n";
            } else {
                out << " += S::mul(w_, " << from->name() << ");\n";
            }
        }
    }
    else {
        if (d.cell_index_var.empty()) {
            switch (constraint) {
                case simd_expr_constraint::contiguous:
                    out << "indirect(" << d.data_var << " + " << index_i_name(d.node_index_var) << ", simd_width_) = ";
                    break;
                case simd_expr_constraint::constant:
                    out << "indirect(" << d.data_var << ", simd_cast<simd_index>(" << index_i_name(d.node_index_var) << "), simd_width_, constraint_category_) = ";
                    break;
                default:
                    out << "indirect(" << d.data_var << ", " << index_i_name(d.node_index_var) << ", simd_width_, constraint_category_) = ";
            }
        } else {
            out << "indirect(" << d.data_var << ", " << index_i_name(d.cell_index_var) << ", simd_width_, index_constraint::none) = ";
        }

        if (coeff != 1) {
            out << "(S::mul(simd_cast<simd_value>(" << as_c_double(coeff) << ")," << from->name() << "));\n";
        } else {
            out << from->name() << ";\n";
        }
    }
}

void emit_simd_index_initialize(std::ostream& out, const std::list<index_prop>& indices,
                           simd_expr_constraint constraint) {
    for (auto& index: indices) {
        if (index.node_index) {
            switch (constraint) {
                case simd_expr_constraint::contiguous:
                case simd_expr_constraint::constant:
                    out << "auto " << index_i_name(index.source_var) << " = " << index.source_var << "[" << index.index_name << "];\n";
                    break;
                default:
                    out << "auto " << index_i_name(index.source_var) << " = simd_cast<simd_index>(indirect(" << index.source_var
                        << ".data() + " << index.index_name << ", simd_width_));\n";
                    break;
            }
        } else {
            switch (constraint) {
                case simd_expr_constraint::contiguous:
                    out << "auto " << index_i_name(index.source_var) << " = simd_cast<simd_index>(indirect(" << index.source_var
                        << " + " << index.index_name << ", simd_width_));\n";
                    break;
                case simd_expr_constraint::constant:
                    out << "auto " << index_i_name(index.source_var) << " = simd_cast<simd_index>(" << index.source_var
                        << "[" << index.index_name << "]);\n";
                    break;
                default:
                    out << "auto " << index_i_name(index.source_var) << " = simd_cast<simd_index>(indirect(" << index.source_var
                        << ", " << index.index_name << ", simd_width_, constraint_category_));\n";
                    break;
            }
        }
    }
}

void emit_simd_body_for_loop(
        std::ostream& out,
        BlockExpression* body,
        const std::vector<LocalVariable*>& indexed_vars,
        const std::vector<VariableExpression*>& scalars,
        const std::list<index_prop>& indices,
        const simd_expr_constraint& constraint) {
    emit_simd_index_initialize(out, indices, constraint);

    for (auto& sym: indexed_vars) {
        emit_simd_state_read(out, sym, constraint);
    }

    simdprint printer(body, scalars);
    printer.set_indirect_index();

    out << printer;

    for (auto& sym: indexed_vars) {
        emit_simd_state_update(out, sym, sym->external_variable(), constraint);
    }
}

void emit_simd_for_loop_per_constraint(std::ostream& out, BlockExpression* body,
                                  const std::vector<LocalVariable*>& indexed_vars,
                                  const std::vector<VariableExpression*>& scalars,
                                  bool requires_weight,
                                  const std::list<index_prop>& indices,
                                  const simd_expr_constraint& constraint,
                                  std::string underlying_constraint_name) {

    out << "constraint_category_ = index_constraint::"<< underlying_constraint_name << ";\n";
    out << "for (unsigned i_ = 0; i_ < index_constraints_." << underlying_constraint_name
        << ".size(); i_++) {\n"
        << indent;

    out << "index_type index_ = index_constraints_." << underlying_constraint_name << "[i_];\n";
    if (requires_weight) {
        out << "simd_value w_;\n"
            << "assign(w_, indirect((weight_+index_), simd_width_));\n";
    }

    emit_simd_body_for_loop(out, body, indexed_vars, scalars, indices, constraint);

    out << popindent << "}\n";
}

void emit_simd_api_body(std::ostream& out, APIMethod* method, const std::vector<VariableExpression*>& scalars) {
    auto body = method->body();
    auto indexed_vars = indexed_locals(method->scope());
    bool requires_weight = false;

    std::vector<LocalVariable*> scalar_indexed_vars;
    std::list<index_prop> indices;

    for (auto& s: body->is_block()->statements()) {
        if (s->is_assignment()) {
            for (auto& v: indexed_vars) {
                if (s->is_assignment()->lhs()->is_identifier()->name() == v->external_variable()->name()) {
                    auto info = decode_indexed_variable(v->external_variable());
                    if (info.accumulate) {
                        requires_weight = true;
                    }
                    break;
                }
            }
        }
    }

    for (auto& sym: indexed_vars) {
        auto info = decode_indexed_variable(sym->external_variable());
        if (!info.scalar()) {
            index_prop node_idx = {info.node_index_var, "index_", true};
            auto it = std::find(indices.begin(), indices.end(), node_idx);
            if (it == indices.end()) indices.push_front(node_idx);

            if (!info.cell_index_var.empty()) {
                index_prop cell_idx = {info.cell_index_var, index_i_name(info.node_index_var), false};
                it = std::find(indices.begin(), indices.end(), cell_idx);
                if (it == indices.end()) indices.push_front(cell_idx);
            }
        }
        else {
            scalar_indexed_vars.push_back(sym);
        }
    }
    if (!body->statements().empty()) {
        out << "assert(simd_width_ <= (unsigned)S::width(simd_cast<simd_value>(0)));\n";
        if (!indices.empty()) {
            out << "index_constraint constraint_category_;\n\n";

            //Generate for loop for all contiguous simd_vectors
            simd_expr_constraint constraint = simd_expr_constraint::contiguous;
            std::string underlying_constraint = "contiguous";

            emit_simd_for_loop_per_constraint(out, body, indexed_vars, scalars, requires_weight, indices, constraint, underlying_constraint);

            //Generate for loop for all independent simd_vectors
            constraint = simd_expr_constraint::other;
            underlying_constraint = "independent";

            emit_simd_for_loop_per_constraint(out, body, indexed_vars, scalars, requires_weight, indices, constraint, underlying_constraint);

            //Generate for loop for all simd_vectors that have no optimizing constraints
            constraint = simd_expr_constraint::other;
            underlying_constraint = "none";

            emit_simd_for_loop_per_constraint(out, body, indexed_vars, scalars, requires_weight, indices, constraint, underlying_constraint);

            //Generate for loop for all constant simd_vectors
            constraint = simd_expr_constraint::constant;
            underlying_constraint = "constant";

            emit_simd_for_loop_per_constraint(out, body, indexed_vars, scalars, requires_weight, indices, constraint, underlying_constraint);

        }
        else {
            // We may nonetheless need to read a global scalar indexed variable.
            for (auto& sym: scalar_indexed_vars) {
                emit_simd_state_read(out, sym, simd_expr_constraint::other);
            }

            out <<
                "unsigned n_ = width_;\n\n"
                "for (unsigned i_ = 0; i_ < n_; i_ += simd_width_) {\n" << indent <<
                simdprint(body, scalars) << popindent <<
                "}\n";
        }
    }
}

