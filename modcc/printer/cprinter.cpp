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
#include "printer/marks.hpp"

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

inline static std::string make_cpu_class_name(const std::string& module_name) { return std::string{"mechanism_cpu_"} + module_name; }

inline static std::string make_cpu_ppack_name(const std::string& module_name) { return make_cpu_class_name(module_name) + std::string{"_pp_"}; }

struct index_prop {
    std::string source_var; // array holding the indices
    std::string index_name; // index into the array
    bool        node_index; // node index (cv) or cell index
    bool operator==(const index_prop& other) const {
        return (source_var == other.source_var) && (index_name == other.index_name);
    }
};

void emit_procedure_proto(std::ostream&, ProcedureExpression*, const std::string&, const std::string& qualified = "");
void emit_simd_procedure_proto(std::ostream&, ProcedureExpression*, const std::string&, const std::string& qualified = "");
void emit_masked_simd_procedure_proto(std::ostream&, ProcedureExpression*, const std::string&, const std::string& qualified = "");

void emit_api_body(std::ostream&, APIMethod*, bool cv_loop = true);
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
    auto name           = module_.module_name();
    auto class_name     = make_cpu_class_name(name);
    auto namespace_name = "kernel_" + class_name;
    auto ppack_name     = make_cpu_ppack_name(name);
    auto ns_components  = namespace_components(opt.cpp_namespace);

    APIMethod* net_receive_api = find_api_method(module_, "net_rec_api");
    APIMethod* post_event_api  = find_api_method(module_, "post_event_api");
    APIMethod* init_api        = find_api_method(module_, "init");
    APIMethod* state_api       = find_api_method(module_, "advance_state");
    APIMethod* current_api     = find_api_method(module_, "compute_currents");
    APIMethod* write_ions_api  = find_api_method(module_, "write_ions");

    bool with_simd = opt.simd.abi!=simd_spec::none;

    options_trace_codegen = opt.trace_codegen;
    
    // init_api, state_api, current_api methods are mandatory:

    assert_has_scope(init_api, "init");
    assert_has_scope(state_api, "advance_state");
    assert_has_scope(current_api, "compute_currents");

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

    ENTER(out);
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

    out << "struct " << ppack_name << ": public ::arb::multicore::mechanism_ppack {\n" << indent;
    for (const auto& scalar: vars.scalars) {
        out << "::arb::fvm_value_type " << scalar->name() <<  " = " << as_c_double(scalar->value()) << ";\n";
    }
    for (const auto& array: vars.arrays) {
        out << "::arb::fvm_value_type* " << array->name() << ";\n";
    }
    for (const auto& dep: ion_deps) {
        out << "::arb::ion_state_view " << ion_state_field(dep.name) << ";\n";
        out << "::arb::fvm_index_type* " << ion_state_index(dep.name) << ";\n";
    }
    out << popindent << "};\n\n";

    // Make implementations
    auto emit_body = [&](APIMethod *p) {
        if (with_simd) {
            emit_simd_api_body(out, p, vars.scalars);
        } else {
            emit_api_body(out, p);
        }
    };

    out << "namespace " << namespace_name << " {\n";

    out << "// procedure prototypes\n";
    for (auto proc: normal_procedures(module_)) {
        if (with_simd) {
            emit_simd_procedure_proto(out, proc, ppack_name);
            out << ";\n";
            emit_masked_simd_procedure_proto(out, proc, ppack_name);
            out << ";\n";
        } else {
            emit_procedure_proto(out, proc, ppack_name);
            out << ";\n";
        }
    }
    out << "\n";

    out << "// interface methods\n";
    out << "void init(" << ppack_name << "* pp) {\n" << indent;
    emit_body(init_api);
    out << popindent << "}\n\n";

    out << "void advance_state(" << ppack_name << "* pp) {\n" << indent;
    out << profiler_enter("advance_integrate_state");
    emit_body(state_api);
    out << profiler_leave();
    out << popindent << "}\n\n";

    out << "void compute_currents(" << ppack_name << "* pp) {\n" << indent;
    out << profiler_enter("advance_integrate_current");
    emit_body(current_api);
    out << profiler_leave();
    out << popindent << "}\n\n";

    out << "void write_ions(" << ppack_name << "* pp) {\n" << indent;
    emit_body(write_ions_api);
    out << popindent << "}\n\n";

    if (net_receive_api) {
        const std::string weight_arg = net_receive_api->args().empty() ? "weight" : net_receive_api->args().front()->is_argument()->name();
        out <<
            "void net_receive(" << ppack_name << "* pp, int i_, ::arb::fvm_value_type " << weight_arg << ") {\n" << indent;
            emit_api_body(out, net_receive_api, false);
            out << popindent <<
            "}\n\n"
            "void apply_events(" << ppack_name << "* pp, ::arb::fvm_size_type mechanism_id, ::arb::multicore::deliverable_event_stream::state events) {\n" << indent <<
            "auto ncell = events.n_streams();\n"
            "for (::arb::fvm_size_type c = 0; c<ncell; ++c) {\n" << indent <<
            "auto begin = events.begin_marked(c);\n"
            "auto end = events.end_marked(c);\n"
            "for (auto p = begin; p<end; ++p) {\n" << indent <<
            "if (p->mech_id==mechanism_id) " << namespace_name << "::net_receive(pp, p->mech_index, p->weight);\n" << popindent <<
            "}\n" << popindent <<
            "}\n" << popindent <<
            "}\n"
            "\n";
    }

    if(post_event_api) {
        const std::string time_arg = post_event_api->args().empty() ? "time" : post_event_api->args().front()->is_argument()->name();
        out <<
            "void post_event(" << ppack_name << "* pp) {\n" << indent <<
            "int n_ = pp->width_;\n"
            "for (int i_ = 0; i_ < n_; ++i_) {\n" << indent <<
            "auto node_index_i_ = pp->node_index_[i_];\n"
            "auto cid_ = pp->vec_ci_[node_index_i_];\n"
            "auto offset_ = pp->n_detectors_ * cid_;\n"
            "for (::arb::fvm_index_type c = 0; c < pp->n_detectors_; c++) {\n" << indent <<
            "auto " << time_arg << " = pp->time_since_spike_[offset_ + c];\n"
            "if (" <<  time_arg << " >= 0) {\n" << indent;
            emit_api_body(out, post_event_api, false);
            out << popindent <<
            "}\n" << popindent <<
            "}\n" << popindent <<
            "}\n" << popindent <<
            "}\n\n";
    }


    out << "// Procedure definitions\n";
    for (auto proc: normal_procedures(module_)) {
        if (with_simd) {
            emit_simd_procedure_proto(out, proc, ppack_name);
            auto simd_print = simdprint(proc->body(), vars.scalars);
            out << " {\n" << indent << simd_print << popindent <<  "}\n\n";

            emit_masked_simd_procedure_proto(out, proc, ppack_name);
            auto masked_print = simdprint(proc->body(), vars.scalars);
            masked_print.set_masked();
            out << " {\n" << indent << masked_print << popindent << "}\n\n";
        } else {
            emit_procedure_proto(out, proc, ppack_name);
            out <<
                " {\n" << indent <<
                cprint(proc->body()) << popindent <<
                "}\n\n";
        }
    }

    out << popindent << "}\n\n"; // close kernel namespace

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
        "void init() override { " << namespace_name << "::init(&pp_); }\n"
        "void advance_state() override { " << namespace_name << "::advance_state(&pp_); }\n"
        "void compute_currents() override { " << namespace_name << "::compute_currents(&pp_); }\n"
        "void write_ions() override{ " << namespace_name << "::write_ions(&pp_); }\n";

    net_receive_api &&
        out << "void apply_events(deliverable_event_stream::state events) override { " << namespace_name << "::apply_events(&pp_, mechanism_id_, events); }\n";

    post_event_api &&
        out << "void post_event() override { " << namespace_name <<  "::post_event(&pp_); };\n";

    with_simd &&
        out << "unsigned simd_width() const override { return simd_width_; }\n";

    out <<
        "\n" << popindent <<
        "protected:\n" << indent <<
        "std::size_t object_sizeof() const override { return sizeof(*this); }\n" <<
        "virtual ::arb::mechanism_ppack* ppack_ptr() override { return &pp_; }\n";

    io::separator sep("\n", ",\n");
    if (!vars.scalars.empty()) {
        out <<
            "mechanism_global_table global_table() override {\n" << indent <<
            "return {" << indent;

        for (const auto& scalar: vars.scalars) {
            auto memb = scalar->name();
            out << sep << "{" << quote(memb) << ", &pp_." << memb << "}";
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
            out << sep << "{" << quote(memb) << ", &pp_." << memb << "}";
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
                out << sep << "{" << quote(memb) << ", &pp_." << memb << "}";
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
            out << sep << "{\"" << dep.name << "\", &pp_." << ion_state_field(dep.name) << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";

        sep.reset();
        out << "mechanism_ion_index_table ion_index_table() override {\n" << indent << "return {" << indent;
        for (const auto& dep: ion_deps) {
            out << sep << "{\"" << dep.name << "\", &pp_." << ion_state_index(dep.name) << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";
    }

    out << popindent << "\n"
        "private:\n" << indent;
    out << ppack_name << " pp_;\n";

    out << popindent <<
        "};\n\n"
        "template <typename B> ::arb::concrete_mech_ptr<B> make_mechanism_" <<name << "();\n"
        "template <> ::arb::concrete_mech_ptr<backend> make_mechanism_" << name << "<backend>() {\n" << indent <<
        "return ::arb::concrete_mech_ptr<backend>(new " << class_name << "());\n" << popindent <<
        "}\n\n";

    out << namespace_declaration_close(ns_components);
    EXIT(out);
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
    out_ << "pp->" << sym->name() << (sym->is_range()? "[i_]": "");
}

void CPrinter::visit(CallExpression* e) {
    out_ << e->name() << "(pp, i_";
    for (auto& arg: e->args()) {
        out_ << ", ";
        arg->accept(this);
    }
    out_ << ")";
}

void CPrinter::visit(BlockExpression* block) {
    ENTERM(out_, "c:block");
    // Only include local declarations in outer-most block.
    if (!block->is_nested()) {
        auto locals = pure_locals(block->scope());
        if (!locals.empty()) {
            out_ << "::arb::fvm_value_type ";
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
    EXITM(out_, "c:block");
}

static std::string index_i_name(const std::string& index_var) {
    return index_var+"i_";
}

void emit_procedure_proto(std::ostream& out, ProcedureExpression* e, const std::string& ppack_name, const std::string& qualified) {
    out << "void " << qualified << (qualified.empty()? "": "::") << e->name() << "(" << ppack_name << "* pp, int i_";
    for (auto& arg: e->args()) {
        out << ", ::arb::fvm_value_type " << arg->is_argument()->name();
    }
    out << ")";
}

namespace {
    // Access through ppack
    std::string data_via_ppack(const indexed_variable_info& i) { return "pp->" + i.data_var; }
    std::string node_index_i_name(const indexed_variable_info& i) { return i.node_index_var + "i_"; }
    std::string source_index_i_name(const index_prop& i) { return i.source_var + "i_"; }
    std::string source_var(const index_prop& i) { return "pp->" + i.source_var; }

    // Convenience I/O wrapper for emitting indexed access to an external variable.

    struct deref {
        indexed_variable_info d;
        deref(indexed_variable_info d): d(d) {}

        friend std::ostream& operator<<(std::ostream& o, const deref& wrap) {
            auto index_var = wrap.d.cell_index_var.empty() ? wrap.d.node_index_var : wrap.d.cell_index_var;
            auto i_name    = index_i_name(index_var);
            index_var = "pp->" + index_var;
            return o << data_via_ppack(wrap.d) << '[' << (wrap.d.scalar() ? "0": i_name) << ']';
        }
    };
}

std::list<index_prop> gather_indexed_vars(const std::vector<LocalVariable*>& indexed_vars, const std::string& index) {
    std::list<index_prop> indices;
    for (auto& sym: indexed_vars) {
        auto d = decode_indexed_variable(sym->external_variable());
        if (!d.scalar()) {
            index_prop node_idx = {d.node_index_var, index, true};
            auto it = std::find(indices.begin(), indices.end(), node_idx);
            if (it == indices.end()) indices.push_front(node_idx);
            if (!d.cell_index_var.empty()) {
                index_prop cell_idx = {d.cell_index_var, node_index_i_name(d), false};
                auto it = std::find(indices.begin(), indices.end(), cell_idx);
                if (it == indices.end()) indices.push_back(cell_idx);
            }
        }
    }
    return indices;
};

void emit_state_read(std::ostream& out, LocalVariable* local) {
    ENTER(out);
    out << "::arb::fvm_value_type " << cprint(local) << " = ";

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
    EXIT(out);
}

void emit_state_update(std::ostream& out, Symbol* from, IndexedVariable* external) {
    if (!external->is_write()) return;
    ENTER(out);
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
        out << "pp->weight_[i_], " << from->name() << ", " << deref(d) << ");\n";
    }
    else {
        out << deref(d) << " = ";
        if (coeff != 1) {
            out << as_c_double(coeff) << '*';
        }
        out << from->name() << ";\n";
    }
    EXIT(out);
}

void emit_api_body(std::ostream& out, APIMethod* method, bool cv_loop) {
    ENTER(out);
    auto body = method->body();
    auto indexed_vars = indexed_locals(method->scope());

    std::list<index_prop> indices = gather_indexed_vars(indexed_vars, "i_");
    if (!body->statements().empty()) {
        cv_loop && out <<
            "int n_ = pp->width_;\n"
            "for (int i_ = 0; i_ < n_; ++i_) {\n" << indent;

        for (auto index: indices) {
            out << "auto " << source_index_i_name(index) << " = " << source_var(index) << "[" << index.index_name << "];\n";
        }

        for (auto& sym: indexed_vars) {
            emit_state_read(out, sym);
        }
        out << cprint(body);

        for (auto& sym: indexed_vars) {
            emit_state_update(out, sym, sym->external_variable());
        }
        cv_loop && out << popindent << "}\n";
    }
    EXIT(out);
}

// SIMD printing:

void SimdPrinter::visit(IdentifierExpression *e) {
    ENTERM(out_, "identifier");
    e->symbol()->accept(this);
    EXITM(out_, "identifier");
}

void SimdPrinter::visit(LocalVariable* sym) {
    ENTERM(out_, "local");
    out_ << sym->name();
    EXITM(out_, "local");
}

void SimdPrinter::visit(VariableExpression *sym) {
    ENTERM(out_, "variable");
    if (sym->is_range()) {
        auto index = is_indirect_? "index_": "i_";
        out_ << "simd_cast<simd_value>(indirect(pp->" << sym->name() << "+" << index << ", simd_width_))";
    }
    else {
        out_ << "pp->" << sym->name();
    }
    EXITM(out_, "variable");
}

void SimdPrinter::visit(AssignmentExpression* e) {
    ENTERM(out_, "assign");
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
        std::string pfx = lhs->is_local_variable() ? "" : "pp->";
        if(is_indirect_)
            out_ << "indirect(" << pfx << lhs->name() << "+index_, simd_width_) = ";
        else
            out_ << "indirect(" << pfx << lhs->name() << "+i_, simd_width_) = ";

        if (!input_mask_.empty())
            out_ << "S::where(" << input_mask_ << ", ";

        if (cast) out_ << "simd_cast<simd_value>(";
        e->rhs()->accept(this);
        if (cast) out_ << ")";

        if (!input_mask_.empty())
            out_ << ")";
    } else {
        std::string pfx = lhs->is_local_variable() ? "" : "pp->";
        out_ << "assign(" << pfx << lhs->name() << ", ";
        if (auto rhs = e->rhs()->is_identifier()) {
            if (auto sym = rhs->symbol()) {
                // We shouldn't call the rhs visitor in this case because it automatically casts indirect expressions
                if (sym->is_variable() && sym->is_variable()->is_range()) {
                    auto index = is_indirect_ ? "index_" : "i_";
                    out_ << "indirect(pp->" << rhs->name() << "+" << index << ", simd_width_))";
                    return;
                }
            }
        }
        e->rhs()->accept(this);
        out_ << ")";
    }
    EXITM(out_, "assign");
}

void SimdPrinter::visit(CallExpression* e) {
    ENTERM(out_, "call");
    if(is_indirect_)
        out_ << e->name() << "(pp, index_";
    else
        out_ << e->name() << "(pp, i_";
    for (auto& arg: e->args()) {
        out_ << ", ";
        arg->accept(this);
    }
    out_ << ")";
    EXITM(out_, "call");
}

void SimdPrinter::visit(BlockExpression* block) {
    // Only include local declarations in outer-most block.
    ENTERM(out_, "block");
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
    EXITM(out_, "block");
}

void emit_simd_procedure_proto(std::ostream& out, ProcedureExpression* e, const std::string& ppack_name, const std::string& qualified) {
    ENTER(out);
    out << "void " << qualified << (qualified.empty()? "": "::") << e->name() << "(" << ppack_name << "* pp, ::arb::fvm_index_type i_";
    for (auto& arg: e->args()) {
        out << ", const simd_value& " << arg->is_argument()->name();
    }
    out << ")";
    EXIT(out);
}

void emit_masked_simd_procedure_proto(std::ostream& out, ProcedureExpression* e, const std::string& ppack_name, const std::string& qualified) {
    ENTER(out);
    out << "void " << qualified << (qualified.empty()? "": "::") << e->name()
    << "(" << ppack_name << "* pp, ::arb::fvm_index_type i_, simd_mask mask_input_";
    for (auto& arg: e->args()) {
        out << ", const simd_value& " << arg->is_argument()->name();
    }
    out << ")";
    EXIT(out);
}

void emit_simd_state_read(std::ostream& out, LocalVariable* local, simd_expr_constraint constraint) {
    ENTER(out);
    out << "simd_value " << local->name();

    if (local->is_read()) {
        auto d = decode_indexed_variable(local->external_variable());
        if (d.scalar()) {
            out << " = simd_cast<simd_value>(pp->" << d.data_var
                << "[0]);\n";
        }
        else {
            if (d.cell_index_var.empty()) {
                switch (constraint) {
                    case simd_expr_constraint::contiguous:
                        out << ";\n"
                            << "assign(" << local->name() << ", indirect(" << data_via_ppack(d)
                            << " + " << node_index_i_name(d) << ", simd_width_));\n";
                        break;
                    case simd_expr_constraint::constant:
                        out << " = simd_cast<simd_value>(" << data_via_ppack(d)
                            << "[" << node_index_i_name(d)  << "]);\n";
                        break;
                    default:
                        out << ";\n"
                            << "assign(" << local->name() << ", indirect(" << data_via_ppack(d)
                            << ", " << node_index_i_name(d) << ", simd_width_, constraint_category_));\n";
                }
            }
            else {
                out << ";\n"
                    << "assign(" << local->name() << ", indirect(" << data_via_ppack(d)
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
    EXIT(out);
}

void emit_simd_state_update(std::ostream& out, Symbol* from, IndexedVariable* external, simd_expr_constraint constraint) {
    if (!external->is_write()) return;

    auto d = decode_indexed_variable(external);
    double coeff = 1./d.scale;

    if (d.readonly) {
        throw compiler_exception("Cannot assign to read-only external state: "+external->to_string());
    }

    ENTER(out);

    if (d.accumulate) {
        if (d.cell_index_var.empty()) {
            switch (constraint) {
                case simd_expr_constraint::contiguous:
                {
                    std::string tempvar = "t_" + external->name();
                    out << "simd_value " << tempvar << ";\n"
                        << "assign(" << tempvar << ", indirect(" << data_via_ppack(d) << " + " << node_index_i_name(d) << ", simd_width_));\n";
                    if (coeff != 1) {
                        out << tempvar << " = S::fma(S::mul(w_, simd_cast<simd_value>(" << as_c_double(coeff) << "))," << from->name() << ", " << tempvar << ");\n";
                    } else {
                        out << tempvar << " = S::fma(w_, " << from->name() << ", " << tempvar << ");\n";
                    }
                    out << "indirect(" << data_via_ppack(d) << " + " << node_index_i_name(d) << ", simd_width_) = " << tempvar << ";\n";
                    break;
                }
                case simd_expr_constraint::constant:
                {
                    out << "indirect(" << data_via_ppack(d) << ", simd_cast<simd_index>(" << node_index_i_name(d) << "), simd_width_, constraint_category_)";
                    if (coeff != 1) {
                        out << " += S::mul(w_, S::mul(simd_cast<simd_value>(" << as_c_double(coeff) << "), " << from->name() << "));\n";
                    } else {
                        out << " += S::mul(w_, " << from->name() << ");\n";
                    }
                    break;
                }
                default :
                {
                    out << "indirect(" << data_via_ppack(d) << ", " << node_index_i_name(d) << ", simd_width_, constraint_category_)";
                    if (coeff != 1) {
                        out << " += S::mul(w_, S::mul(simd_cast<simd_value>(" << as_c_double(coeff) << "), " << from->name() << "));\n";
                    } else {
                        out << " += S::mul(w_, " << from->name() << ");\n";
                    }
                }
            }
        } else {
            out << "indirect(" << data_via_ppack(d) << ", " << index_i_name(d.cell_index_var) << ", simd_width_, index_constraint::none)";
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
                    out << "indirect(" << data_via_ppack(d) << " + " << node_index_i_name(d) << ", simd_width_) = ";
                    break;
                case simd_expr_constraint::constant:
                    out << "indirect(" << data_via_ppack(d) << ", simd_cast<simd_index>(" << node_index_i_name(d) << "), simd_width_, constraint_category_) = ";
                    break;
                default:
                    out << "indirect(" << data_via_ppack(d) << ", " << node_index_i_name(d) << ", simd_width_, constraint_category_) = ";
            }
        } else {
            out << "indirect(" << data_via_ppack(d)



                << ", " << index_i_name(d.cell_index_var) << ", simd_width_, index_constraint::none) = ";
        }

        if (coeff != 1) {
            out << "(S::mul(simd_cast<simd_value>(" << as_c_double(coeff) << ")," << from->name() << "));\n";
        } else {
            out << from->name() << ";\n";
        }
    }

    EXIT(out);
}

void emit_simd_index_initialize(std::ostream& out, const std::list<index_prop>& indices,
                                simd_expr_constraint constraint) {
    ENTER(out);
    for (auto& index: indices) {
        if (index.node_index) {
            switch (constraint) {
                case simd_expr_constraint::contiguous:
                case simd_expr_constraint::constant:
                    out << "auto " << source_index_i_name(index) << " = " << source_var(index) << "[" << index.index_name << "];\n";
                    break;
                default:
                    out << "auto " << source_index_i_name(index) << " = simd_cast<simd_index>(indirect(&" << source_var(index)
                        << "[0] + " << index.index_name << ", simd_width_));\n";
                    break;
            }
        } else {
            switch (constraint) {
                case simd_expr_constraint::contiguous:
                    out << "auto " << source_index_i_name(index) << " = simd_cast<simd_index>(indirect(" << source_var(index)
                        << " + " << index.index_name << ", simd_width_));\n";
                    break;
                case simd_expr_constraint::constant:
                    out << "auto " << source_index_i_name(index) << " = simd_cast<simd_index>(" << source_var(index)
                        << "[" << index.index_name << "]);\n";
                    break;
                default:
                    out << "auto " << source_index_i_name(index) << " = simd_cast<simd_index>(indirect(" << source_var(index)
                        << ", " << index.index_name << ", simd_width_, constraint_category_));\n";
                    break;
            }
        }
    }
    EXIT(out);
}

void emit_simd_body_for_loop(
        std::ostream& out,
        BlockExpression* body,
        const std::vector<LocalVariable*>& indexed_vars,
        const std::vector<VariableExpression*>& scalars,
        const std::list<index_prop>& indices,
        const simd_expr_constraint& constraint) {
    ENTER(out);
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
    EXIT(out);
}

void emit_simd_for_loop_per_constraint(std::ostream& out, BlockExpression* body,
                                  const std::vector<LocalVariable*>& indexed_vars,
                                  const std::vector<VariableExpression*>& scalars,
                                  bool requires_weight,
                                  const std::list<index_prop>& indices,
                                  const simd_expr_constraint& constraint,
                                  std::string underlying_constraint_name) {
    ENTER(out);
    out << "constraint_category_ = index_constraint::"<< underlying_constraint_name << ";\n";
    out << "for (unsigned i_ = 0; i_ < pp->index_constraints_." << underlying_constraint_name
        << ".size(); i_++) {\n"
        << indent;

    out << "::arb::fvm_index_type index_ = pp->index_constraints_." << underlying_constraint_name << "[i_];\n";
    if (requires_weight) {
        out << "simd_value w_;\n"
            << "assign(w_, indirect((pp->weight_+index_), simd_width_));\n";
    }

    emit_simd_body_for_loop(out, body, indexed_vars, scalars, indices, constraint);

    out << popindent << "}\n";
    EXIT(out);
}

void emit_simd_api_body(std::ostream& out, APIMethod* method, const std::vector<VariableExpression*>& scalars) {
    auto body = method->body();
    auto indexed_vars = indexed_locals(method->scope());
    bool requires_weight = false;

    ENTER(out);

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
    std::list<index_prop> indices = gather_indexed_vars(indexed_vars, "index_");
    std::vector<LocalVariable*> scalar_indexed_vars;
    for (auto& sym: indexed_vars) {
        if (decode_indexed_variable(sym->external_variable()).scalar()) {
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
                "unsigned n_ = pp->width_;\n\n"
                "for (unsigned i_ = 0; i_ < n_; i_ += simd_width_) {\n" << indent <<
                simdprint(body, scalars) << popindent <<
                "}\n";
        }
    }
    EXIT(out);
}
