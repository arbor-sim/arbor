#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>

#include "cudaprinter.hpp"
#include "expression.hpp"
#include "io/ostream_wrappers.hpp"
#include "io/prefixbuf.hpp"
#include "printer/cexpr_emit.hpp"
#include "printer/printerutil.hpp"

using io::indent;
using io::popindent;
using io::quote;

void emit_common_defs(std::ostream&, const Module& module_);
void emit_api_body_cu(std::ostream& out, APIMethod* method, bool is_point_proc);
void emit_procedure_body_cu(std::ostream& out, ProcedureExpression* proc);
void emit_state_read_cu(std::ostream& out, LocalVariable* local);
void emit_state_update_cu(std::ostream& out, Symbol* from,
                          IndexedVariable* external, bool is_point_proc);
const char* index_id(Symbol *s);

struct cuprint {
    Expression* expr_;
    explicit cuprint(Expression* expr): expr_(expr) {}

    friend std::ostream& operator<<(std::ostream& out, const cuprint& w) {
        CudaPrinter printer(out);
        return w.expr_->accept(&printer), out;
    }
};

std::string make_class_name(const std::string& module_name) {
    return "mechanism_gpu_"+module_name;
}

std::string make_ppack_name(const std::string& module_name) {
    return make_class_name(module_name)+"_pp_";
}

static std::string ion_state_field(const std::string& ion_name) {
    return "ion_"+ion_name+"_";
}

static std::string ion_state_index(const std::string& ion_name) {
    return "ion_"+ion_name+"_index_";
}

std::string emit_cuda_cpp_source(const Module& module_, const printer_options& opt) {
    std::string name = module_.module_name();
    std::string class_name = make_class_name(name);
    std::string ppack_name = make_ppack_name(name);
    auto ns_components = namespace_components(opt.cpp_namespace);

    NetReceiveExpression* net_receive = find_net_receive(module_);

    auto vars = local_module_variables(module_);
    auto ion_deps = module_.ion_deps();

    std::string fingerprint = "<placeholder>";

    io::pfxstringstream out;

    net_receive && out <<
        "#include <" << arb_private_header_prefix() << "backends/event.hpp>\n"
        "#include <" << arb_private_header_prefix() << "backends/multi_event_stream_state.hpp>\n";

    out <<
        "#include <" << arb_private_header_prefix() << "backends/gpu/mechanism.hpp>\n"
        "#include <" << arb_private_header_prefix() << "backends/gpu/mechanism_ppack_base.hpp>\n";

    out << "\n" << namespace_declaration_open(ns_components) << "\n";

    emit_common_defs(out, module_);

    out <<
        "void " << class_name << "_nrn_init_(" << ppack_name << "&);\n"
        "void " << class_name << "_nrn_state_(" << ppack_name << "&);\n"
        "void " << class_name << "_nrn_current_(" << ppack_name << "&);\n"
        "void " << class_name << "_write_ions_(" << ppack_name << "&);\n";

    net_receive && out <<
        "void " << class_name << "_deliver_events_(int mech_id, "
        << ppack_name << "&, deliverable_event_stream_state events);\n";

    out <<
        "\n"
        "class " << class_name << ": public ::arb::gpu::mechanism {\n"
        "public:\n" << indent <<
        "const ::arb::mechanism_fingerprint& fingerprint() const override {\n" << indent <<
        "static ::arb::mechanism_fingerprint hash = " << quote(fingerprint) << ";\n"
        "return hash;\n" << popindent <<
        "}\n\n"
        "std::string internal_name() const override { return " << quote(name) << "; }\n"
        "::arb::mechanismKind kind() const override { return " << module_kind_str(module_) << "; }\n"
        "::arb::mechanism_ptr clone() const override { return ::arb::mechanism_ptr(new " << class_name << "()); }\n"
        "\n"
        "void nrn_init() override {\n" << indent <<
        class_name << "_nrn_init_(pp_);\n" << popindent <<
        "}\n\n"
        "void nrn_state() override {\n" << indent <<
        class_name << "_nrn_state_(pp_);\n" << popindent <<
        "}\n\n"
        "void nrn_current() override {\n" << indent <<
        class_name << "_nrn_current_(pp_);\n" << popindent <<
        "}\n\n"
        "void write_ions() override {\n" << indent <<
        class_name << "_write_ions_(pp_);\n" << popindent <<
        "}\n\n";

    net_receive && out <<
        "void deliver_events(deliverable_event_stream_state events) override {\n" << indent <<
        class_name << "_deliver_events_(mechanism_id_, pp_, events);\n" << popindent <<
        "}\n\n";

    out << popindent <<
        "protected:\n" << indent <<
        "std::size_t object_sizeof() const override { return sizeof(*this); }\n"
        "::arb::gpu::mechanism_ppack_base* ppack_ptr() { return &pp_; }\n\n";

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
        "private:\n" << indent <<
        make_ppack_name(name) << " pp_;\n" << popindent <<
        "};\n\n"
        "template <typename B> ::arb::concrete_mech_ptr<B> make_mechanism_" << name << "();\n"
        "template <> ::arb::concrete_mech_ptr<::arb::gpu::backend> make_mechanism_" << name << "<::arb::gpu::backend>() {\n" << indent <<
        "return ::arb::concrete_mech_ptr<::arb::gpu::backend>(new " << class_name << "());\n" << popindent <<
        "}\n\n";

    out << namespace_declaration_close(ns_components);
    return out.str();
}

std::string emit_cuda_cu_source(const Module& module_, const printer_options& opt) {
    std::string name = module_.module_name();
    std::string class_name = make_class_name(name);
    std::string ppack_name = make_ppack_name(name);
    auto ns_components = namespace_components(opt.cpp_namespace);
    const bool is_point_proc = module_.kind() == moduleKind::point;

    NetReceiveExpression* net_receive = find_net_receive(module_);
    APIMethod* init_api = find_api_method(module_, "nrn_init");
    APIMethod* state_api = find_api_method(module_, "nrn_state");
    APIMethod* current_api = find_api_method(module_, "nrn_current");
    APIMethod* write_ions_api = find_api_method(module_, "write_ions");

    assert_has_scope(init_api, "nrn_init");
    assert_has_scope(state_api, "nrn_state");
    assert_has_scope(current_api, "nrn_current");

    io::pfxstringstream out;

    out <<
        "#include <iostream>\n"
        "#include <" << arb_private_header_prefix() << "backends/event.hpp>\n"
        "#include <" << arb_private_header_prefix() << "backends/multi_event_stream_state.hpp>\n"
        "#include <" << arb_private_header_prefix() << "backends/gpu/cuda_common.hpp>\n"
        "#include <" << arb_private_header_prefix() << "backends/gpu/math_cu.hpp>\n"
        "#include <" << arb_private_header_prefix() << "backends/gpu/mechanism_ppack_base.hpp>\n";

    is_point_proc && out <<
        "#include <" << arb_private_header_prefix() << "backends/gpu/reduce_by_key.hpp>\n";

    out << "\n" << namespace_declaration_open(ns_components) << "\n";

    out <<
        "using value_type = ::arb::gpu::mechanism_ppack_base::value_type;\n"
        "using index_type = ::arb::gpu::mechanism_ppack_base::index_type;\n"
        "\n";

    emit_common_defs(out, module_);

    // Print the CUDA code and kernels:
    //  - first __device__ functions that implement NMODL PROCEDUREs.
    //  - then __global__ kernels that implement API methods and call the procedures.

    out << "namespace {\n\n"; // place inside an anonymous namespace

    out << "using ::arb::gpu::exprelr;\n";
    out << "using ::arb::gpu::safeinv;\n";
    out << "using ::arb::gpu::min;\n";
    out << "using ::arb::gpu::max;\n\n";

    // Procedures as __device__ functions.
    auto emit_procedure_kernel = [&] (ProcedureExpression* e) {
        out << "__device__\n"
            << "void " << e->name()
            << "(" << ppack_name << " params_, int tid_";
        for(auto& arg: e->args()) {
            out << ", value_type " << arg->is_argument()->name();
        }
        out << ") {\n" << indent
            << cuprint(e->body())
            << popindent << "}\n\n";
    };

    for (auto& p: module_normal_procedures(module_)) {
        emit_procedure_kernel(p);
    }

    // API methods as __global__ kernels.
    auto emit_api_kernel = [&] (APIMethod* e) {
        // Only print the kernel if the method is not empty.
        if (!e->body()->statements().empty()) {
            out << "__global__\n"
                << "void " << e->name() << "(" << ppack_name << " params_) {\n" << indent
                << "int n_ = params_.width_;\n";
            emit_api_body_cu(out, e, is_point_proc);
            out << popindent << "}\n\n";
        }
    };

    emit_api_kernel(init_api);
    emit_api_kernel(state_api);
    emit_api_kernel(current_api);
    emit_api_kernel(write_ions_api);

    // event delivery
    if (net_receive) {
        out << "__global__\n"
            << "void deliver_events(int mech_id_, " <<  ppack_name << " params_, "
            << "deliverable_event_stream_state events) {\n" << indent
            << "auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n"
            << "auto const ncell_ = events.n;\n\n"

            << "if(tid_<ncell_) {\n" << indent
            << "auto begin = events.ev_data+events.begin_offset[tid_];\n"
            << "auto end = events.ev_data+events.end_offset[tid_];\n"
            << "for (auto p = begin; p<end; ++p) {\n" << indent
            << "if (p->mech_id==mech_id_) {\n" << indent
            << "auto tid_ = p->mech_index;\n"
            << "auto weight = p->weight;\n"
            << cuprint(net_receive->body())
            << popindent << "}\n"
            << popindent << "}\n"
            << popindent << "}\n"
            << popindent << "}\n";
    }

    out << "} // namspace\n\n"; // close anonymous namespace

    // Write wrappers.
    auto emit_api_wrapper = [&] (APIMethod* e) {
        out << "void " << class_name << "_" << e->name() << "_(" << ppack_name << "& p) {";

        // Only call the kernel if the kernel is required.
        !e->body()->statements().empty() && out
            << "\n" << indent
            << "auto n = p.width_;\n"
            << "unsigned block_dim = 128;\n"
            << "unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);\n"
            << e->name() << "<<<grid_dim, block_dim>>>(p);\n"
            << popindent;

        out << "}\n\n";
    };
    emit_api_wrapper(init_api);
    emit_api_wrapper(current_api);
    emit_api_wrapper(state_api);
    emit_api_wrapper(write_ions_api);

    net_receive && out
        << "void " << class_name << "_deliver_events_("
        << "int mech_id, "
        << ppack_name << "& p, deliverable_event_stream_state events) {\n" << indent
        << "auto n = events.n;\n"
        << "unsigned block_dim = 128;\n"
        << "unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);\n"
        << "deliver_events<<<grid_dim, block_dim>>>(mech_id, p, events);\n"
        << popindent << "}\n\n";

    out << namespace_declaration_close(ns_components);
    return out.str();
}

void emit_common_defs(std::ostream& out, const Module& module_) {
    std::string class_name = make_class_name(module_.module_name());
    std::string ppack_name = make_ppack_name(module_.module_name());

    auto vars = local_module_variables(module_);
    auto ion_deps = module_.ion_deps();

    find_net_receive(module_) && out <<
        "using deliverable_event_stream_state =\n"
        "    ::arb::multi_event_stream_state<::arb::deliverable_event_data>;\n\n";

    out << "struct " << ppack_name << ": ::arb::gpu::mechanism_ppack_base {\n" << indent;

    for (const auto& scalar: vars.scalars) {
        out << "value_type " << scalar->name() <<  " = " << as_c_double(scalar->value()) << ";\n";
    }
    for (const auto& array: vars.arrays) {
        out << "value_type* " << array->name() << ";\n";
    }
    for (const auto& dep: ion_deps) {
        out << "ion_state_view " << ion_state_field(dep.name) << ";\n";
        out << "const index_type* " << ion_state_index(dep.name) << ";\n";
    }

    out << popindent << "};\n\n";
}

static std::string index_i_name(const std::string& index_var) {
    return index_var+"i_";
}

void emit_api_body_cu(std::ostream& out, APIMethod* e, bool is_point_proc) {
    auto body = e->body();
    auto indexed_vars = indexed_locals(e->scope());

    std::unordered_set<std::string> indices;
    for (auto& sym: indexed_vars) {
        auto d = decode_indexed_variable(sym->external_variable());
        if (!d.scalar()) {
            indices.insert(d.index_var);
        }
    }

    if (!body->statements().empty()) {
        out << "int tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n";
        if (is_point_proc) {
            // The run length information is only required if this method will
            // update an indexed variable, like current or conductance.
            // This is the case if one of the external variables "is_write".
            auto it = std::find_if(indexed_vars.begin(), indexed_vars.end(),
                      [](auto& sym){return sym->external_variable()->is_write();});
            if (it!=indexed_vars.end()) {
                out << "unsigned lane_mask_ = __ballot_sync(0xffffffff, tid_<n_);\n";
            }
        }

        out << "if (tid_<n_) {\n" << indent;

        for (auto& index: indices) {
            out << "auto " << index_i_name(index)
                << " = params_." << index << "[tid_];\n";
        }

        for (auto& sym: indexed_vars) {
            emit_state_read_cu(out, sym);
        }

        out << cuprint(body);

        for (auto& sym: indexed_vars) {
            emit_state_update_cu(out, sym, sym->external_variable(), is_point_proc);
        }
        out << popindent << "}\n";
    }
}

void emit_procedure_body_cu(std::ostream& out, ProcedureExpression* e) {
    out << cuprint(e->body());
}

namespace {
    // Convenience I/O wrapper for emitting indexed access to an external variable.

    struct deref {
        indexed_variable_info v;

        deref(indexed_variable_info v): v(v) {}
        friend std::ostream& operator<<(std::ostream& o, const deref& wrap) {
            return o << "params_." << wrap.v.data_var << '['
                     << (wrap.v.scalar()? "0": index_i_name(wrap.v.index_var)) << ']';
        }
    };
}

void emit_state_read_cu(std::ostream& out, LocalVariable* local) {
    out << "value_type " << cuprint(local) << " = ";

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


void emit_state_update_cu(std::ostream& out, Symbol* from,
                          IndexedVariable* external, bool is_point_proc)
{
    if (!external->is_write()) return;

    auto d = decode_indexed_variable(external);
    double coeff = 1./d.scale;

    if (d.readonly) {
        throw compiler_exception("Cannot assign to read-only external state: "+external->to_string());
    }

    if (is_point_proc && d.accumulate) {
        out << "::arb::gpu::reduce_by_key(";
        if (coeff != 1) out << as_c_double(coeff) << '*';

        out << "params_.weight_[tid_]*" << from->name() << ',';
        out << "params_." << d.data_var << ", " << index_i_name(d.index_var) << ", lane_mask_);\n";
    }
    else if (d.accumulate) {
        out << deref(d) << " = fma(";
        if (coeff != 1) out << as_c_double(coeff) << '*';

        out << "params_.weight_[tid_], " << from->name() << ", " << deref(d) << ");\n";
    }
    else {
        out << deref(d) << " = ";
        if (coeff != 1) out << as_c_double(coeff) << '*';

        out << from->name() << ";\n";
    }
}

// CUDA Printer visitors

void CudaPrinter::visit(VariableExpression *sym) {
    out_ << "params_." << sym->name() << (sym->is_range()? "[tid_]": "");
}

void CudaPrinter::visit(CallExpression* e) {
    out_ << e->name() << "(params_, tid_";
    for (auto& arg: e->args()) {
        out_ << ", ";
        arg->accept(this);
    }
    out_ << ")";
}
