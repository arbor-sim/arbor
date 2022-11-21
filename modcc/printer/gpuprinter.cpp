#include <cmath>
#include <iostream>
#include <string>
#include <set>
#include <regex>

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

static std::string scaled(double coeff) {
    std::stringstream ss;
    if (coeff != 1) {
        ss << as_c_double(coeff) << '*';
    }
    return ss.str();
}


void emit_api_body_cu(std::ostream& out, APIMethod* method, const ApiFlags&);
void emit_state_read_cu(std::ostream& out, LocalVariable* local, const ApiFlags&);
void emit_state_update_cu(std::ostream& out, Symbol* from, IndexedVariable* external, const ApiFlags&);

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
static std::string ion_field(const IonDep& ion) { return fmt::format("ion_{}",       ion.name); }
static std::string ion_index(const IonDep& ion) { return fmt::format("ion_{}_index", ion.name); }


ARB_LIBMODCC_API std::string emit_gpu_cpp_source(const Module& module_, const printer_options& opt) {
    std::string name       = module_.module_name();
    std::string class_name = make_class_name(name);
    std::string ppack_name = make_ppack_name(name);
    auto ns_components     = namespace_components(opt.cpp_namespace);
    auto vars = local_module_variables(module_);
    auto ion_deps = module_.ion_deps();

    io::pfxstringstream out;

    out << "#include <arbor/mechanism_abi.h>\n"
        << "#include <cmath>\n\n"
        << namespace_declaration_open(ns_components)
        << fmt::format("void {0}_init_(arb_mechanism_ppack*);\n"
                       "void {0}_advance_state_(arb_mechanism_ppack*);\n"
                       "void {0}_compute_currents_(arb_mechanism_ppack*);\n"
                       "void {0}_write_ions_(arb_mechanism_ppack*);\n"
                       "void {0}_apply_events_(arb_mechanism_ppack*, arb_deliverable_event_stream*);\n"
                       "void {0}_post_event_(arb_mechanism_ppack*);\n\n",
                       class_name)
        << namespace_declaration_close(ns_components)
        << "\n";

    std::stringstream ss;
    for (const auto& c: ns_components) ss << c <<"::";

    out << fmt::format(FMT_COMPILE("extern \"C\" {{\n"
                                   "  arb_mechanism_interface* make_{4}_{1}_interface_gpu() {{\n"
                                   "    static arb_mechanism_interface result;\n"
                                   "    result.backend={2};\n"
                                   "    result.partition_width=1;\n"
                                   "    result.alignment=1;\n"
                                   "    result.init_mechanism={3}{0}_init_;\n"
                                   "    result.compute_currents={3}{0}_compute_currents_;\n"
                                   "    result.apply_events={3}{0}_apply_events_;\n"
                                   "    result.advance_state={3}{0}_advance_state_;\n"
                                   "    result.write_ions={3}{0}_write_ions_;\n"
                                   "    result.post_event={3}{0}_post_event_;\n"
                                   "    return &result;\n"
                                   "  }}\n"
                                   "}};\n\n"),
                       class_name,
                       name,
                       "arb_backend_kind_gpu",
                       ss.str(),
                       std::regex_replace(opt.cpp_namespace, std::regex{"::"}, "_"));
    EXIT(out);
    return out.str();
}

ARB_LIBMODCC_API std::string emit_gpu_cu_source(const Module& module_, const printer_options& opt) {
    std::string name = module_.module_name();
    std::string class_name = make_class_name(name);

    auto ns_components = namespace_components(opt.cpp_namespace);

    const bool is_point_proc = (module_.kind() == moduleKind::point) || (module_.kind() == moduleKind::junction);

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

    auto vars = local_module_variables(module_);

    out << "#include <arbor/gpu/gpu_common.hpp>\n"
           "#include <arbor/gpu/math_cu.hpp>\n"
           "#include <arbor/gpu/reduce_by_key.hpp>\n"
           "#include <arbor/mechanism_abi.h>\n";

    out << "\n" << namespace_declaration_open(ns_components) << "\n";

    out << fmt::format(FMT_COMPILE("#define PPACK_IFACE_BLOCK \\\n"
                                   "auto  {0}width             __attribute__((unused)) = params_.width;\\\n"
                                   "auto  {0}n_detectors       __attribute__((unused)) = params_.n_detectors;\\\n"
                                   "auto* {0}vec_ci            __attribute__((unused)) = params_.vec_ci;\\\n"
                                   "auto* {0}vec_di            __attribute__((unused)) = params_.vec_di;\\\n"
                                   "auto* {0}vec_dt            __attribute__((unused)) = params_.vec_dt;\\\n"
                                   "auto* {0}vec_v             __attribute__((unused)) = params_.vec_v;\\\n"
                                   "auto* {0}vec_i             __attribute__((unused)) = params_.vec_i;\\\n"
                                   "auto* {0}vec_g             __attribute__((unused)) = params_.vec_g;\\\n"
                                   "auto* {0}temperature_degC  __attribute__((unused)) = params_.temperature_degC;\\\n"
                                   "auto* {0}diam_um           __attribute__((unused)) = params_.diam_um;\\\n"
                                   "auto* {0}time_since_spike  __attribute__((unused)) = params_.time_since_spike;\\\n"
                                   "auto* {0}node_index        __attribute__((unused)) = params_.node_index;\\\n"
                                   "auto* {0}peer_index        __attribute__((unused)) = params_.peer_index;\\\n"
                                   "auto* {0}multiplicity      __attribute__((unused)) = params_.multiplicity;\\\n"
                                   "auto* {0}state_vars        __attribute__((unused)) = params_.state_vars;\\\n"
                                   "auto* {0}weight            __attribute__((unused)) = params_.weight;\\\n"
                                   "auto& {0}events            __attribute__((unused)) = params_.events;\\\n"
                                   "auto& {0}mechanism_id      __attribute__((unused)) = params_.mechanism_id;\\\n"
                                   "auto& {0}index_constraints __attribute__((unused)) = params_.index_constraints;\\\n"),
                       pp_var_pfx);

    const auto& [state_ids, global_ids, param_ids, white_noise_ids] = public_variable_ids(module_);
    const auto& assigned_ids = module_.assigned_block().parameters;

    auto global = 0;
    for (const auto& scalar: global_ids) {
        out << fmt::format("auto {}{} __attribute__((unused)) = params_.globals[{}];\\\n", pp_var_pfx, scalar.name(), global);
        global++;
    }
    out << fmt::format("auto const * const * {}random_numbers  __attribute__((unused)) = params_.random_numbers;\\\n", pp_var_pfx);
    auto param = 0, state = 0;
    for (const auto& array: state_ids) {
        out << fmt::format("auto* {}{} __attribute__((unused)) = params_.state_vars[{}];\\\n", pp_var_pfx, array.name(), state);
        state++;
    }
    for (const auto& array: assigned_ids) {
        out << fmt::format("auto* {}{} __attribute__((unused)) = params_.state_vars[{}];\\\n", pp_var_pfx, array.name(), state);
        state++;
    }
    for (const auto& array: param_ids) {
        out << fmt::format("auto* {}{} __attribute__((unused)) = params_.parameters[{}];\\\n", pp_var_pfx, array.name(), param);
        param++;
    }
    auto idx = 0;
    for (const auto& ion: module_.ion_deps()) {
        out << fmt::format("auto& {}{} __attribute__((unused)) = params_.ion_states[{}];\\\n",       pp_var_pfx, ion_field(ion), idx);
        out << fmt::format("auto* {}{} __attribute__((unused)) = params_.ion_states[{}].index;\\\n", pp_var_pfx, ion_index(ion), idx);
        idx++;
    }
    out << "//End of IFACEBLOCK\n\n";

    // Print the CUDA code and kernels:
    //  - first __device__ functions that implement NMODL PROCEDUREs.
    //  - then __global__ kernels that implement API methods and call the procedures.

    out << "namespace {\n\n" // place inside an anonymous namespace
        << "using ::arb::gpu::exprelr;\n"
        << "using ::arb::gpu::safeinv;\n"
        << "using ::arb::gpu::min;\n"
        << "using ::arb::gpu::max;\n\n";

    // API methods as __global__ kernels.
    auto emit_api_kernel = [&] (APIMethod* e, bool additive=false) {
        // Only print the kernel if the method is not empty.
        if (!e->body()->statements().empty()) {
            out << "__global__\n"
                << "void " << e->name() << "(arb_mechanism_ppack params_) {\n" << indent
                << "int n_ = params_.width;\n"
                << "int tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n";
            emit_api_body_cu(out, e, ApiFlags{}.point(is_point_proc).additive(additive).voltage(moduleKind::voltage == module_.kind()));
            out << popindent << "}\n\n";
        }
    };

    emit_api_kernel(init_api);
    if (init_api && !init_api->body()->statements().empty()) {
        out << fmt::format(FMT_COMPILE("__global__\n"
                                       "void multiply(arb_mechanism_ppack params_) {{\n"
                                       "    PPACK_IFACE_BLOCK;\n"
                                       "    auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n"
                                       "    auto idx_ = blockIdx.y;"
                                       "    if(tid_<{0}width) {{\n"
                                       "        {0}state_vars[idx_][tid_] *= {0}multiplicity[tid_];\n"
                                       "    }}\n"
                                       "}}\n\n"),
                           pp_var_pfx);
    }
    emit_api_kernel(state_api);
    emit_api_kernel(current_api, true);
    emit_api_kernel(write_ions_api);

    // event delivery
    if (net_receive_api) {
        out << fmt::format(FMT_COMPILE("__global__\n"
                                       "void apply_events(arb_mechanism_ppack params_, arb_deliverable_event_stream stream) {{\n"
                                       "    PPACK_IFACE_BLOCK;\n"
                                       "    auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n"
                                       "    if(tid_<stream.n_streams) {{\n"
                                       "        auto begin = stream.events + stream.begin[tid_];\n"
                                       "        auto end   = stream.events + stream.end[tid_];\n"
                                       "        for (auto p = begin; p<end; ++p) {{\n"
                                       "            if (p->mech_id=={1}mechanism_id) {{\n"
                                       "                auto tid_ = p->mech_index;\n"
                                       "                auto {0} = p->weight;\n"),
                           net_receive_api->args().empty() ? "weight" : net_receive_api->args().front()->is_argument()->name(),
                           pp_var_pfx);
        out << indent << indent << indent << indent;
        emit_api_body_cu(out, net_receive_api, ApiFlags{}.point(is_point_proc).loop(false).iface(false));
        out << popindent << "}\n" << popindent << "}\n" << popindent << "}\n" << popindent << "}\n";
    }

    // event delivery
    if (post_event_api) {
        const std::string time_arg = post_event_api->args().empty() ? "time" : post_event_api->args().front()->is_argument()->name();
        out << fmt::format(FMT_COMPILE("__global__\n"
                                       "void post_event(arb_mechanism_ppack params_) {{\n"
                                       "    PPACK_IFACE_BLOCK;\n"
                                       "    auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;\n"
                                       "    if (tid_<{1}width) {{\n"
                                       "        auto node_index_i_ = {1}node_index[tid_];\n"
                                       "        auto cid_ = {1}vec_ci[node_index_i_];\n"
                                       "        auto offset_ = {1}n_detectors * cid_;\n"
                                       "        for (unsigned c = 0; c < {1}n_detectors; c++) {{\n"
                                       "            auto {0} = {1}time_since_spike[offset_ + c];\n"
                                       "            if ({0} >= 0) {{\n"),
                           time_arg,
                           pp_var_pfx);
        out << indent << indent << indent << indent;
        emit_api_body_cu(out, post_event_api, ApiFlags{}.point(is_point_proc).loop(false).iface(false));
        out << popindent << "}\n" << popindent << "}\n" << popindent << "}\n" << popindent << "}\n";
    }

    out << "} // namespace\n\n"; // close anonymous namespace

    // Write wrappers.
    auto emit_api_wrapper = [&] (APIMethod* e, const auto& width, std::string_view name="") {
        auto api_name = name.empty() ? e->name() : name;
        out << fmt::format(FMT_COMPILE("void {}_{}_(arb_mechanism_ppack* p) {{"), class_name, api_name);
        if(!e->body()->statements().empty()) {
            out << fmt::format(FMT_COMPILE("\n"
                                           "    auto n = p->{};\n"
                                           "    unsigned block_dim = 128;\n"
                                           "    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);\n"
                                           "    {}<<<grid_dim, block_dim>>>(*p);\n"),
                               width,
                               api_name);
        }
        out << "}\n\n";
    };

    auto emit_empty_wrapper = [&] (std::string_view name) {
        out << fmt::format(FMT_COMPILE("void {}_{}_(arb_mechanism_ppack* p) {{}}\n"), class_name, name);
    };


    {
        auto api_name = init_api->name();
        auto n = std::count_if(vars.arrays.begin(), vars.arrays.end(),
                               [] (const auto& v) { return v->is_state(); });

        out << fmt::format(FMT_COMPILE("void {}_{}_(arb_mechanism_ppack* p) {{"), class_name, api_name);
        if(!init_api->body()->statements().empty()) {
            out << fmt::format(FMT_COMPILE("\n"
                                           "    auto n = p->{0};\n"
                                           "    unsigned block_dim = 128;\n"
                                           "    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);\n"
                                           "    {1}<<<grid_dim, block_dim>>>(*p);\n"
                                           "    if (!p->multiplicity) return;\n"
                                           "    multiply<<<dim3{{grid_dim, {2}}}, block_dim>>>(*p);\n"),
                               "width",
                               api_name,
                               n);
        }
        out << "}\n\n";
    }

    emit_api_wrapper(current_api,     "width");
    emit_api_wrapper(state_api,       "width");
    emit_api_wrapper(write_ions_api,  "width");
    if (post_event_api) {
        emit_api_wrapper(post_event_api, "width", "post_event");
    } else {
        emit_empty_wrapper("post_event");
    }
    if (net_receive_api) {
        auto api_name = "apply_events";
        out << fmt::format(FMT_COMPILE("void {}_{}_(arb_mechanism_ppack* p, arb_deliverable_event_stream* stream_ptr) {{"), class_name, api_name);
        if(!net_receive_api->body()->statements().empty()) {
            out << fmt::format(FMT_COMPILE("\n"
                                           "    auto n = stream_ptr->n_streams;\n"
                                           "    unsigned block_dim = 128;\n"
                                           "    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);\n"
                                           "    {}<<<grid_dim, block_dim>>>(*p, *stream_ptr);\n"),
                               api_name);
        }
        out << "}\n\n";
    } else {
        auto api_name = "apply_events";
        out << fmt::format(FMT_COMPILE("void {}_{}_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {{}}\n\n"), class_name, api_name);
    }
    out << namespace_declaration_close(ns_components);
    return out.str();
}

static std::string index_i_name(const std::string& index_var) {
    return index_var+"i_";
}

void emit_api_body_cu(std::ostream& out, APIMethod* e, const ApiFlags& flags) {
    auto body = e->body();
    auto indexed_vars = indexed_locals(e->scope());

    struct index_prop {
        std::string source_var; // array holding the indices
        std::string index_name; // index into the array
        bool operator==(const index_prop& other) const {
            return (source_var == other.source_var) && (index_name==other.index_name);
        }
    };

    // Gather the indices that need to be read at the beginning
    // of an APIMethod in the order that they should be read
    // eg:
    //   node_index_ = node_index[tid];
    //   domain_index_ = vec_di[node_index_];
    std::list<index_prop> indices;
    for (auto& sym: indexed_vars) {
        auto d = decode_indexed_variable(sym->external_variable());
        if (!d.scalar()) {
            auto nested = !d.inner_index_var().empty();
            if (nested) {
                // Need to read 2 indices: outer[inner[tid]]
                index_prop inner_index_prop = {d.inner_index_var(), "tid_"};
                index_prop outer_index_prop = {d.outer_index_var(), index_i_name(d.inner_index_var())};

                // Check that the outer and inner indices haven't already been added to the list
                auto inner_it = std::find(indices.begin(), indices.end(), inner_index_prop);
                auto outer_it = std::find(indices.begin(), indices.end(), outer_index_prop);

                // The inner index needs to be read before the outer index
                if (inner_it == indices.end()) {
                    indices.push_front(inner_index_prop);
                }
                if (outer_it == indices.end()) {
                    indices.push_back(outer_index_prop);
                }
            }
            else {
                // Need to read 1 index: outer[index]
                index_prop outer_index_prop = {d.outer_index_var(), "tid_"};

                // Check that the index hasn't already been added to the list
                auto it = std::find(indices.begin(), indices.end(), outer_index_prop);
                if (it == indices.end()) {
                    indices.push_front(outer_index_prop);
                }
            }
        }
    }

    if (!body->statements().empty()) {
        if (flags.is_point) {
            // The run length information is only required if this method will
            // update an indexed variable, like current or conductance.
            // This is the case if one of the external variables "is_write".
            auto it = std::find_if(indexed_vars.begin(), indexed_vars.end(),
                      [](auto& sym){return sym->external_variable()->is_write();});
            if (it!=indexed_vars.end()) {
                out << "unsigned lane_mask_ = arb::gpu::ballot(0xffffffff, tid_<n_);\n";
            }
        }
        if (flags.ppack_iface) out << "PPACK_IFACE_BLOCK;\n";
        if (flags.cv_loop) out << "if (tid_<n_) {\n" << indent;

        for (auto& index: indices) {
            out << "auto " << index_i_name(index.source_var)
                << " = " << pp_var_pfx << index.source_var << "[" << index.index_name << "];\n";
        }

        for (auto& sym: indexed_vars) {
            emit_state_read_cu(out, sym, flags);
        }

        out << cuprint(body);

        for (auto& sym: indexed_vars) {
            emit_state_update_cu(out, sym, sym->external_variable(), flags);
        }
        if (flags.cv_loop) out << popindent << "}\n";
    }
}

namespace {
    // Convenience I/O wrapper for emitting indexed access to an external variable.

    struct deref {
        indexed_variable_info v;

        deref(indexed_variable_info v): v(v) {}
        friend std::ostream& operator<<(std::ostream& o, const deref& wrap) {
            auto index_var = wrap.v.outer_index_var();
            return o << pp_var_pfx << wrap.v.data_var << '['
                     << (wrap.v.scalar()? "0": index_i_name(index_var)) << ']';
        }
    };
}

void emit_state_read_cu(std::ostream& out, LocalVariable* local, const ApiFlags& flags) {
    auto write_voltage = local->external_variable()->data_source() == sourceKind::voltage
                      && flags.can_write_voltage;
    out << "arb_value_type " << cuprint(local) << " = ";
    auto d = decode_indexed_variable(local->external_variable());
    if (local->is_read() || (local->is_write() && d.additive) || write_voltage) {
        if (d.scale != 1) {
            out << as_c_double(d.scale) << "*";
        }
        out << deref(d) << ";\n";
    }
    else {
        out << "0;\n";
    }
}


void emit_state_update_cu(std::ostream& out,
                          Symbol* from,
                          IndexedVariable* external,
                          const ApiFlags& flags) {
    if (!external->is_write()) return;
    auto d = decode_indexed_variable(external);
    auto write_voltage = external->data_source() == sourceKind::voltage
                      && flags.can_write_voltage;
    if (write_voltage) d.readonly = false;
    if (d.readonly) {
        throw compiler_exception("Cannot assign to read-only external state: "+external->to_string());
    }

    auto name   = from->name();
    auto scale  = scaled(1.0/d.scale);
    auto data   = pp_var_pfx + d.data_var;
    auto index  = index_i_name(d.outer_index_var());
    auto var    = deref(d);
    auto use_weight = d.always_use_weight || !flags.is_point;
    std::string weight = scale + (use_weight ? pp_var_pfx + "weight[tid_]" : "1.0");

    if (d.additive && flags.use_additive) {
        out << name << " -= " << var << ";\n";
        if (flags.is_point) {
            out << fmt::format("::arb::gpu::reduce_by_key({}*{}, {}, {}, lane_mask_);\n", weight, name, data, index);
        }
        else {
            out << var << " = fma(" << weight << ", " << name << ", " << var << ");\n";
        }
    }
    else if (write_voltage) {
        /* SAFETY:
        ** - Only one V-PROCESS per CV
        ** - these can never be point mechs
        ** - they run separatly from density/point mechs
        */
        out << name << " -= " << var << ";\n"
            << var << " = fma(" << weight << ", " << name << ", " << var << ");\n";
    }
    else if (d.accumulate) {
        if (flags.is_point) {
            out << "::arb::gpu::reduce_by_key(" << weight << "*" << name << ',' << data << ", " << index << ", lane_mask_);\n";
        }
        else {
            out << var << " = fma(" << weight << ", " << name << ", " << var << ");\n";
        }
    }
    else {
        out << var << " = " << scale << name << ";\n";
    }
}

// CUDA Printer visitors

void GpuPrinter::visit(VariableExpression *sym) {
    out_ << pp_var_pfx << sym->name() << (sym->is_range()? "[tid_]": "");
}

void GpuPrinter::visit(CallExpression* e) {
    out_ << e->name() << "(params_, tid_";
    for (auto& arg: e->args()) {
        out_ << ", ";
        arg->accept(this);
    }
    out_ << ")";
}

void GpuPrinter::visit(WhiteNoise* sym) {
    out_ << fmt::format("{}random_numbers[{}][tid_]", pp_var_pfx, sym->index());
}
