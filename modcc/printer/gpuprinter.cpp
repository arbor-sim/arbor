#include <cmath>
#include <iostream>
#include <string>
#include <set>
#include <regex>

#define FMT_HEADER_ONLY YES
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/compile.h>

#include "gpuprinter.hpp"
#include "expression.hpp"
#include "io/ostream_wrappers.hpp"
#include "io/prefixbuf.hpp"
#include "printer/cexpr_emit.hpp"
#include "printer/printerutil.hpp"

using io::indent;
using io::popindent;
using io::quote;

void emit_api_body_cu(std::ostream& out, APIMethod* method, bool is_point_proc, bool cv_loop = true);
void emit_procedure_body_cu(std::ostream& out, ProcedureExpression* proc);
void emit_state_read_cu(std::ostream& out, LocalVariable* local);
void emit_state_update_cu(std::ostream& out, Symbol* from, IndexedVariable* external, bool is_point_proc);

const char* index_id(Symbol *s);

struct cuprint {
    Expression* expr_;
    explicit cuprint(Expression* expr): expr_(expr) {}

    friend std::ostream& operator<<(std::ostream& out, const cuprint& w) {
        GpuPrinter printer(out);
        return w.expr_->accept(&printer), out;
    }
};

static std::string make_class_name(const std::string& n) { return "mechanism_" + n + "_gpu";}
static std::string make_ppack_name(const std::string& module_name) { return make_class_name(module_name)+"_pp_"; }
static std::string ion_state_field(const std::string& ion_name) { return "ion_"+ion_name+"_"; }
static std::string ion_state_index(const std::string& ion_name) { return "ion_"+ion_name+"_index_"; }

std::string emit_gpu_cpp_source(const Module& module_, const printer_options& opt) {
    std::string name       = module_.module_name();
    std::string class_name = make_class_name(name);
    auto namespace_name = "kernel_" + name;
    std::string ppack_name = make_ppack_name(name);
    auto ns_components     = namespace_components(opt.cpp_namespace);
    std::string fingerprint = "<placeholder>";
    auto vars = local_module_variables(module_);
    auto ion_deps = module_.ion_deps();

    io::pfxstringstream out;

    out << "#include <arbor/mechanism_abi.h>\n"
        << "#include <cmath>\n\n"
        << namespace_declaration_open(ns_components)
        << "namespace " << namespace_name << " {\n"
        << fmt::format("void {0}_init_(arb_mechanism_ppack&);\n"
                       "void {0}_advance_state_(arb_mechanism_ppack&);\n"
                       "void {0}_compute_currents_(arb_mechanism_ppack&);\n"
                       "void {0}_write_ions_(arb_mechanism_ppack&);\n"
                       "void {0}_apply_events_(arb_mechanism_ppack&);\n"
                       "void {0}_post_event_(arb_mechanism_ppack&);\n\n"
                       "void init(arb_mechanism_ppack* pp)             {{ {0}_init_(*pp); }}\n"
                       "void advance_state(arb_mechanism_ppack* pp)    {{ {0}_advance_state_(*pp); }}\n"
                       "void compute_currents(arb_mechanism_ppack* pp) {{ {0}_compute_currents_(*pp); }}\n"
                       "void write_ions(arb_mechanism_ppack* pp)       {{ {0}_write_ions_(*pp); }}\n"
                       "void apply_events(arb_mechanism_ppack* pp)     {{ {0}_apply_events_(*pp); }}\n"
                       "void post_event(arb_mechanism_ppack* pp)       {{ {0}_post_event_(*pp);  }}\n",
                       class_name)
        << "} // " << namespace_name << "\n\n"
        << "// Tables\n";
    {
        auto n = 0ul;
        io::separator sep("", ", ");
        out << "static const char* globals[]               = { ";
        for (const auto& var: vars.scalars) {
            out << sep << quote(var->name());
            ++n;
        }
        out << " };\n"
            << "static arb_size_type n_globals             = " << n << ";\n";
    }

    {
        io::separator sep("", ", ");
        out << "static arb_value_type global_defaults[]    = { ";
        for (const auto& var: vars.scalars) {
            out << sep << (std::isnan(var->value()) ? "NAN" : std::to_string(var->value()));
        }
        out << " };\n";
    }

    {
        auto n = 0ul;
        io::separator sep("", ", ");
        out << "static const char* state_vars[]            = { ";
        for (const auto& var: vars.arrays) {
            if(var->is_state()) {
                out << sep << quote(var->name());
                ++n;
            }
        }
        out << " };\n"
            << "static arb_size_type n_state_vars          = " << n << ";\n";
    }

    {
        io::separator sep("", ", ");
        out << "static arb_value_type state_var_defaults[] = { ";
        for (const auto& var: vars.arrays) {
            if(var->is_state()) {
                out << sep << (std::isnan(var->value()) ? "NAN" : std::to_string(var->value()));
            }
        }
        out << " };\n";
    }

    {
        auto n = 0ul;
        io::separator sep("", ", ");
        out << "static const char* parameters[]            = { ";
        for (const auto& var: vars.arrays) {
            if(!var->is_state()) {
                out << sep << quote(var->name());
                ++n;
            }
        }
        out << " };\n"
            << "static arb_size_type n_parameters          = " << n << ";\n";
    }

    {
        io::separator sep("", ", ");
        out << "static arb_value_type parameter_defaults[] = { ";
        for (const auto& var: vars.arrays) {
            if(!var->is_state()) {
                out << sep << (std::isnan(var->value()) ? "NAN" : std::to_string(var->value()));
            }
        }
        out << " };\n";
    }

    {
        io::separator sep("", ", ");
        out << "static const char* ions[]                  = { ";
        auto n = 0ul;
        for (const auto& dep: ion_deps) {
            out << sep << quote(dep.name);
            ++n;
        }
        out << " };\n"
            << "static arb_size_type n_ions                = " << n << ";\n";
    }

    out << fmt::format("\n// GPU Interface\n"
                       "static arb_mechanism_interface iface_{2}_gpu {{\n"
                       "    .backend={0},\n"
                       "    .init_mechanism=(arb_mechanism_method){1}::init,\n"
                       "    .compute_currents=(arb_mechanism_method){1}::compute_currents,\n"
                       "    .apply_events=(arb_mechanism_method){1}::apply_events,\n"
                       "    .advance_state=(arb_mechanism_method){1}::advance_state,\n"
                       "    .write_ions=(arb_mechanism_method){1}::write_ions,\n"
                       "    .post_event=(arb_mechanism_method){1}::post_event\n"
                       "}};\n\n",
                       "arb_backend_kind::cpu",
                       namespace_name,
                       name)
        << fmt::format("// Mechanism plugin\n"
                       "static arb_mechanism_type {0} {{\n"
                       "   .abi_version=ARB_MECH_ABI_VERSION,\n"
                       "   .fingerprint=\"{1}\",\n"
                       "   .name=\"{0}\",\n"
                       "   .kind={2},\n"
                       "   .partition_width=0,\n"
                       "   .globals=globals,\n"
                       "   .global_defaults=global_defaults,\n"
                       "   .n_globals=n_globals,\n"
                       "   .ions=ions,\n"
                       "   .n_ions=n_ions,\n"
                       "   .state_vars=state_vars,\n"
                       "   .state_var_defaults=state_var_defaults,\n"
                       "   .n_state_vars=n_state_vars,\n"
                       "   .parameters=parameters,\n"
                       "   .parameter_defaults=parameter_defaults,\n"
                       "   .n_parameters=n_parameters,\n"
                       "   .interface=&iface_{0}_gpu\n"
                       "}};\n"
                       "\n",
                       name,
                       fingerprint,
                       module_kind_str(module_))
        << namespace_declaration_close(ns_components)
        << "\n"
        << fmt::format("arb_mechanism_type* make_{0}_{1}() {{ return &{2}::{1}; }}\n",
                       std::regex_replace(opt.cpp_namespace, std::regex{"::"}, "_"),
                       name,
                       opt.cpp_namespace);
    EXIT(out);
    return out.str();
}

std::string emit_gpu_cu_source(const Module& module_, const printer_options& opt) {
    std::string name = module_.module_name();
    std::string class_name = make_class_name(name);

    auto ns_components = namespace_components(opt.cpp_namespace);

    const bool is_point_proc = module_.kind() == moduleKind::point;

    APIMethod* net_receive_api = find_api_method(module_, "net_rec_api");
    APIMethod* post_event_api  = find_api_method(module_, "post_event_api");
    APIMethod* init_api        = find_api_method(module_, "init");
    APIMethod* state_api       = find_api_method(module_, "advance_state");
    APIMethod* current_api     = find_api_method(module_, "compute_currents");
    APIMethod* write_ions_api  = find_api_method(module_, "write_ions");

    assert_has_scope(init_api,    "init");
    assert_has_scope(state_api,   "advance_state");
    assert_has_scope(current_api, "compute_currents");

    io::pfxstringstream out;

    out << fmt::format(FMT_COMPILE("#include <{0}backends/gpu/gpu_common.hpp>\n"
                                   "#include <{0}backends/gpu/math_cu.hpp>\n"
                                   "#include <{0}backends/gpu/reduce_by_key.hpp>\n"
                                   "#include <arbor/mechanism_abi.h>\n"),
                       arb_private_header_prefix());

    out << "\n" << namespace_declaration_open(ns_components) << "\n";
    // Print the CUDA code and kernels:
    //  - first __device__ functions that implement NMODL PROCEDUREs.
    //  - then __global__ kernels that implement API methods and call the procedures.

    out << "namespace {\n\n" // place inside an anonymous namespace
        << "using ::arb::gpu::exprelr;\n"
        << "using ::arb::gpu::safeinv;\n"
        << "using ::arb::gpu::min;\n"
        << "using ::arb::gpu::max;\n\n";

    // Procedures as __device__ functions.
    auto emit_procedure_kernel = [&] (ProcedureExpression* e) {
        out << fmt::format("__device__\n"
                           "void {}(arb_mechanism_ppack params_, int tid_",
                           e->name());
        for(auto& arg: e->args()) out << ", arb_value_type " << arg->is_argument()->name();
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
                << "void " << e->name() << "(arb_mechanism_ppack params_) {\n" << indent
                << "int n_ = params_.width;\n"
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
        out << fmt::format(FMT_COMPILE("__global__\n"
                                       "void apply_events(arb_mechanism_ppack params_) {{\n"
                                       "  auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n"
                                       "  auto events = params_events;\n"
                                       "  auto const ncell_ = events.n;\n"
                                       "  if(tid_<ncell_) {{\n"
                                       "    auto begin = events.ev_data+events.begin_offset[tid_];\n"
                                       "    auto end = events.ev_data+events.end_offset[tid_];\n"
                                       "    for (auto p = begin; p<end; ++p) {{\n"
                                       "      if (p->mech_id==params_.mechanism_id) {{\n"
                                       "        auto tid_ = p->mech_index;\n"
                                       "        auto {0} = p->weight;\n"),
                           net_receive_api->args().empty() ? "weight" : net_receive_api->args().front()->is_argument()->name());
        out << indent << indent << indent << indent;
        emit_api_body_cu(out, net_receive_api, is_point_proc, false);
        out << popindent << "}\n" << popindent << "}\n" << popindent << "}\n" << popindent << "}\n";
    }

    // event delivery
    if (post_event_api) {
        const std::string time_arg = post_event_api->args().empty() ? "time" : post_event_api->args().front()->is_argument()->name();
        out << fmt::format(FMT_COMPILE("__global__\n"
                                       "void post_event(arb_mechanism_ppack* params_) {{\n"
                                       "  int n_ = params_.width_;\n"
                                       "  auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n"
                                       "  if (tid_<n_) {{\n"
                                       "    auto node_index_i_ = params_.node_index_[tid_];\n"
                                       "    auto cid_ = params_.vec_ci_[node_index_i_];\n"
                                       "    auto offset_ = params_.n_detectors_ * cid_;\n"
                                       "    for (unsigned c = 0; c < params_.n_detectors_; c++) {{\n"
                                       "      auto {0} = params_.time_since_spike_[offset_ + c];\n"
                                       "      if ({0} >= 0) {{\n"),
                          time_arg);
        out << indent << indent << indent << indent;
        emit_api_body_cu(out, post_event_api, is_point_proc, false);
        out << popindent << "}\n" << popindent << "}\n" << popindent << "}\n" << popindent << "}\n";
    }

    out << "} // namespace\n\n"; // close anonymous namespace

    // Write wrappers.
    auto emit_api_wrapper = [&] (APIMethod* e, const auto& width, std::string_view name="") {
        auto api_name = name.empty() ? e->name() : name;
        out << fmt::format(FMT_COMPILE("void {}_{}_(arb_mechanism_ppack& p) {{"), class_name, api_name);
        if(!e->body()->statements().empty()) {
            out << fmt::format(FMT_COMPILE("\n"
                                           "  auto n = p.{};\n"
                                           "  unsigned block_dim = 128;\n"
                                           "  unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);\n"
                                           "  {}<<<grid_dim, block_dim>>>(p);\n"),
                               width,
                               api_name);
        }
        out << "}\n\n";
    };

    if (init_api)        emit_api_wrapper(init_api,        "width");
    if (current_api)     emit_api_wrapper(current_api,     "width");
    if (state_api)       emit_api_wrapper(state_api,       "width");
    if (write_ions_api)  emit_api_wrapper(write_ions_api,  "width");
    if (post_event_api)  emit_api_wrapper(post_event_api,  "width");
    if (net_receive_api) emit_api_wrapper(net_receive_api, "events.n", "apply_events");
    out << namespace_declaration_close(ns_components);
    return out.str();
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
    out << "arb_value_type " << cuprint(local) << " = ";

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
                          IndexedVariable* external, bool is_point_proc) {
    if (!external->is_write()) return;

    auto d = decode_indexed_variable(external);
    double coeff = 1./d.scale;

    if (d.readonly) {
        throw compiler_exception("Cannot assign to read-only external state: "+external->to_string());
    }

    if (is_point_proc && d.accumulate) {
        out << "::arb::gpu::reduce_by_key(";
        if (coeff != 1) out << as_c_double(coeff) << '*';

        out << "params_.weight[tid_]*" << from->name() << ',';

        auto index_var = d.cell_index_var.empty() ? d.node_index_var : d.cell_index_var;
        out << "params_." << d.data_var << ", " << index_i_name(index_var) << ", lane_mask_);\n";
    }
    else if (d.accumulate) {
        out << deref(d) << " = fma(";
        if (coeff != 1) out << as_c_double(coeff) << '*';

        out << "params_.weight[tid_], " << from->name() << ", " << deref(d) << ");\n";
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
