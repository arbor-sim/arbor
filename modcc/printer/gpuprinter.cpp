#include <cmath>
#include <iostream>
#include <string>
#include <set>

#include "gpuprinter.hpp"
#include "expression.hpp"
#include "io/ostream_wrappers.hpp"
#include "io/prefixbuf.hpp"
#include "printer/cexpr_emit.hpp"
#include "printer/printerutil.hpp"

using io::indent;
using io::popindent;
using io::quote;

void emit_common_defs(std::ostream&, const Module& module_);
void emit_api_body_cu(std::ostream& out, APIMethod* method, bool is_point_proc, bool cv_loop = true);
void emit_procedure_body_cu(std::ostream& out, ProcedureExpression* proc);
void emit_state_read_cu(std::ostream& out, LocalVariable* local);
void emit_state_update_cu(std::ostream& out, Symbol* from,
                          IndexedVariable* external, bool is_point_proc);
const char* index_id(Symbol *s);

struct cuprint {
    Expression* expr_;
    explicit cuprint(Expression* expr): expr_(expr) {}

    friend std::ostream& operator<<(std::ostream& out, const cuprint& w) {
        GpuPrinter printer(out);
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

std::string emit_gpu_cpp_source(const Module& module_, const printer_options& opt) {
    std::string name = module_.module_name();
    std::string class_name = make_class_name(name);
    std::string ppack_name = make_ppack_name(name);
    auto ns_components = namespace_components(opt.cpp_namespace);

    NetReceiveExpression* net_receive = find_net_receive(module_);
    PostEventExpression*  post_event  = find_post_event(module_);

    auto vars = local_module_variables(module_);
    auto ion_deps = module_.ion_deps();

    std::string fingerprint = "<placeholder>";

    io::pfxstringstream out;

    net_receive && out <<
        "#include <" << arb_private_header_prefix() << "backends/event.hpp>\n"
        "#include <" << arb_private_header_prefix() << "backends/multi_event_stream_state.hpp>\n";

    out << "#include <" << arb_private_header_prefix() << "backends/gpu/mechanism.hpp>\n"
        << "#include <arbor/mechanism_ppack.hpp>\n";

    out << "\n" << namespace_declaration_open(ns_components) << "\n";

    emit_common_defs(out, module_);

    out <<
        "void " << class_name << "_init_(" << ppack_name << "&);\n"
        "void " << class_name << "_advance_state_(" << ppack_name << "&);\n"
        "void " << class_name << "_compute_currents_(" << ppack_name << "&);\n"
        "void " << class_name << "_write_ions_(" << ppack_name << "&);\n";

    net_receive && out <<
        "void " << class_name << "_apply_events_(int mech_id, "
        << ppack_name << "&, deliverable_event_stream_state events);\n";

    post_event && out <<
        "void " << class_name << "_post_event_(" << ppack_name << "&);\n";


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
        "void init() override {\n" << indent <<
        class_name << "_init_(pp_);\n" << popindent <<
        "}\n\n"
        "void advance_state() override {\n" << indent <<
        class_name << "_advance_state_(pp_);\n" << popindent <<
        "}\n\n"
        "void compute_currents() override {\n" << indent <<
        class_name << "_compute_currents_(pp_);\n" << popindent <<
        "}\n\n"
        "void write_ions() override {\n" << indent <<
        class_name << "_write_ions_(pp_);\n" << popindent <<
        "}\n\n";

    net_receive && out <<
        "void apply_events(deliverable_event_stream_state events) override {\n" << indent <<
        class_name << "_apply_events_(mechanism_id_, pp_, events);\n" << popindent <<
        "}\n\n";

    post_event && out <<
        "void post_event() override {\n" << indent <<
        class_name << "_post_event_(pp_);\n" << popindent <<
        "}\n\n";

    out << popindent <<
        "protected:\n" << indent <<
        "std::size_t object_sizeof() const override { return sizeof(*this); }\n"
        "::arb::mechanism_ppack* ppack_ptr() override { return &pp_; }\n\n";

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

std::string emit_gpu_cu_source(const Module& module_, const printer_options& opt) {
    std::string name = module_.module_name();
    std::string class_name = make_class_name(name);
    std::string ppack_name = make_ppack_name(name);
    auto ns_components = namespace_components(opt.cpp_namespace);
    const bool is_point_proc = module_.kind() == moduleKind::point;

    APIMethod* net_receive_api = find_api_method(module_, "net_rec_api");
    APIMethod* post_event_api = find_api_method(module_, "post_event_api");
    APIMethod* init_api = find_api_method(module_, "init");
    APIMethod* state_api = find_api_method(module_, "advance_state");
    APIMethod* current_api = find_api_method(module_, "compute_currents");
    APIMethod* write_ions_api = find_api_method(module_, "write_ions");

    assert_has_scope(init_api, "init");
    assert_has_scope(state_api, "advance_state");
    assert_has_scope(current_api, "compute_currents");

    io::pfxstringstream out;

    out <<
        "#include <iostream>\n"
        "#include <" << arb_private_header_prefix() << "backends/event.hpp>\n"
        "#include <" << arb_private_header_prefix() << "backends/multi_event_stream_state.hpp>\n"
        "#include <" << arb_private_header_prefix() << "backends/gpu/gpu_common.hpp>\n"
        "#include <" << arb_private_header_prefix() << "backends/gpu/math_cu.hpp>\n"
        "#include <arbor/mechanism.hpp>\n" <<
        "#include <arbor/mechanism_ppack.hpp>\n";

    is_point_proc && out <<
        "#include <" << arb_private_header_prefix() << "backends/gpu/reduce_by_key.hpp>\n";

    out << "\n" << namespace_declaration_open(ns_components) << "\n";

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
            out << ", ::arb::fvm_value_type " << arg->is_argument()->name();
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
                << "int n_ = params_.width_;\n"
                << "int tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n";
            emit_api_body_cu(out, e, is_point_proc);
            out << popindent << "}\n\n";
        }
    };

    emit_api_kernel(init_api);
    emit_api_kernel(state_api);
    emit_api_kernel(current_api);
    emit_api_kernel(write_ions_api);

    // event delivery
    if (net_receive_api) {
        const std::string weight_arg = net_receive_api->args().empty() ? "weight" : net_receive_api->args().front()->is_argument()->name();
        out << "__global__\n"
            << "void apply_events(int mech_id_, " <<  ppack_name << " params_, "
            << "deliverable_event_stream_state events) {\n" << indent
            << "auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n"
            << "auto const ncell_ = events.n;\n\n"

            << "if(tid_<ncell_) {\n" << indent
            << "auto begin = events.ev_data+events.begin_offset[tid_];\n"
            << "auto end = events.ev_data+events.end_offset[tid_];\n"
            << "for (auto p = begin; p<end; ++p) {\n" << indent
            << "if (p->mech_id==mech_id_) {\n" << indent
            << "auto tid_ = p->mech_index;\n"
            << "auto " << weight_arg << " = p->weight;\n";
            emit_api_body_cu(out, net_receive_api, is_point_proc, false);
            out << popindent << "}\n"
            << popindent << "}\n"
            << popindent << "}\n"
            << popindent << "}\n";
    }

    // event delivery
    if (post_event_api) {
        const std::string time_arg = post_event_api->args().empty() ? "time" : post_event_api->args().front()->is_argument()->name();
        out << "__global__\n"
            << "void post_event(" <<  ppack_name << " params_) {\n" << indent
            << "int n_ = params_.width_;\n"
            << "auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n"
            << "if (tid_<n_) {\n" << indent
            << "auto node_index_i_ = params_.node_index_[tid_];\n"
            << "auto cid_ = params_.vec_ci_[node_index_i_];\n"
            << "auto offset_ = params_.n_detectors_ * cid_;\n"
            << "for (unsigned c = 0; c < params_.n_detectors_; c++) {\n" << indent
            << "auto " << time_arg << " = params_.time_since_spike_[offset_ + c];\n"
            << "if (" <<  time_arg << " >= 0) {\n" << indent;
            emit_api_body_cu(out, post_event_api, is_point_proc, false);
            out << popindent << "}\n"
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

    net_receive_api && out
        << "void " << class_name << "_apply_events_("
        << "int mech_id, "
        << ppack_name << "& p, deliverable_event_stream_state events) {\n" << indent
        << "auto n = events.n;\n"
        << "unsigned block_dim = 128;\n"
        << "unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);\n"
        << "apply_events<<<grid_dim, block_dim>>>(mech_id, p, events);\n"
        << popindent << "}\n\n";

    post_event_api && out
        << "void " << class_name << "_post_event_("
        << ppack_name << "& p) {\n" << indent
        << "auto n = p.width_;\n"
        << "unsigned block_dim = 128;\n"
        << "unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);\n"
        << "post_event<<<grid_dim, block_dim>>>(p);\n"
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

    out << "struct " << ppack_name << ": ::arb::mechanism_ppack {\n" << indent;

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
}

static std::string index_i_name(const std::string& index_var) {
    return index_var+"i_";
}

void emit_api_body_cu(std::ostream& out, APIMethod* e, bool is_point_proc, bool cv_loop) {
    auto body = e->body();
    auto indexed_vars = indexed_locals(e->scope());

    struct index_prop {
        std::string source_var; // array holding the indices
        std::string index_name; // index into the array
        bool operator==(const index_prop& other) const {
            return (source_var == other.source_var) && (index_name==other.index_name);
        }
    };

    std::list<index_prop> indices;
    for (auto& sym: indexed_vars) {
        auto d = decode_indexed_variable(sym->external_variable());
        if (!d.scalar()) {
            index_prop node_idx = {d.node_index_var, "tid_"};
            auto it = std::find(indices.begin(), indices.end(), node_idx);
            if (it == indices.end()) indices.push_front(node_idx);
            if (!d.cell_index_var.empty()) {
                index_prop cell_idx = {d.cell_index_var, index_i_name(d.node_index_var)};
                auto it = std::find(indices.begin(), indices.end(), cell_idx);
                if (it == indices.end()) indices.push_back(cell_idx);
            }
        }
    }

    if (!body->statements().empty()) {
        if (is_point_proc) {
            // The run length information is only required if this method will
            // update an indexed variable, like current or conductance.
            // This is the case if one of the external variables "is_write".
            auto it = std::find_if(indexed_vars.begin(), indexed_vars.end(),
                      [](auto& sym){return sym->external_variable()->is_write();});
            if (it!=indexed_vars.end()) {
                out << "unsigned lane_mask_ = arb::gpu::ballot(0xffffffff, tid_<n_);\n";
            }
        }

        cv_loop && out << "if (tid_<n_) {\n" << indent;

        for (auto& index: indices) {
            out << "auto " << index_i_name(index.source_var)
                << " = params_." << index.source_var << "[" << index.index_name << "];\n";
        }

        for (auto& sym: indexed_vars) {
            emit_state_read_cu(out, sym);
        }

        out << cuprint(body);

        for (auto& sym: indexed_vars) {
            emit_state_update_cu(out, sym, sym->external_variable(), is_point_proc);
        }
        cv_loop && out << popindent << "}\n";
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
            auto index_var = wrap.v.cell_index_var.empty() ? wrap.v.node_index_var : wrap.v.cell_index_var;
            return o << "params_." << wrap.v.data_var << '['
                     << (wrap.v.scalar()? "0": index_i_name(index_var)) << ']';
        }
    };
}

void emit_state_read_cu(std::ostream& out, LocalVariable* local) {
    out << "::arb::fvm_value_type " << cuprint(local) << " = ";

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

        auto index_var = d.cell_index_var.empty() ? d.node_index_var : d.cell_index_var;
        out << "params_." << d.data_var << ", " << index_i_name(index_var) << ", lane_mask_);\n";
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

void GpuPrinter::visit(VariableExpression *sym) {
    out_ << "params_." << sym->name() << (sym->is_range()? "[tid_]": "");
}

void GpuPrinter::visit(CallExpression* e) {
    out_ << e->name() << "(params_, tid_";
    for (auto& arg: e->args()) {
        out_ << ", ";
        arg->accept(this);
    }
    out_ << ")";
}
