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

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/compile.h>

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

static std::string ion_field(const IonDep& ion) { return fmt::format("ion_{}",       ion.name); }
static std::string ion_index(const IonDep& ion) { return fmt::format("ion_{}_index", ion.name); }

static std::string scaled(double coeff) {
    std::stringstream ss;
    if (coeff != 1) {
        ss << as_c_double(coeff) << '*';
    }
    return ss.str();
}

struct index_prop {
    std::string source_var; // array holding the indices
    std::string index_name; // index into the array
    index_kind  kind; // node index (cv) or cell index or other
    bool operator==(const index_prop& other) const {
        return (source_var == other.source_var) && (index_name == other.index_name);
    }
};

void emit_api_body(std::ostream&, APIMethod*, const ApiFlags& flags={});
void emit_simd_api_body(std::ostream&, APIMethod*, const std::vector<VariableExpression*>& scalars, const ApiFlags&);
void emit_simd_index_initialize(std::ostream& out, const std::list<index_prop>& indices, simd_expr_constraint constraint);

void emit_simd_body_for_loop(std::ostream& out,
                             BlockExpression* body,
                             const std::vector<LocalVariable*>& indexed_vars,
                             const std::list<index_prop>& indices,
                             const simd_expr_constraint& constraint,
                             const ApiFlags&);

void emit_simd_for_loop_per_constraint(std::ostream& out, BlockExpression* body,
                                       const std::vector<LocalVariable*>& indexed_vars,
                                       const std::list<index_prop>& indices,
                                       const simd_expr_constraint& constraint,
                                       std::string constraint_name,
                                       const ApiFlags&);

struct cprint {
    Expression* expr_;
    explicit cprint(Expression* expr): expr_(expr) {}

    friend std::ostream& operator<<(std::ostream& out, const cprint& w) {
        CPrinter printer(out);
        return w.expr_->accept(&printer), out;
    }
};

std::string do_cprint(Expression* cp, int ind) {
    std::stringstream ss;
    for (auto i = 0; i < ind; ++i) ss << indent;
    ss << cprint(cp);
    for (auto i = 0; i < ind; ++i) ss << popindent;
    return ss.str();
}

struct simdprint {
    Expression* expr_;
    bool is_indirect_ = false; // For choosing between "index_" and "i_" as an index. Depends on whether
                               // we are in a procedure or handling a simd constraint in an API call.
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

ARB_LIBMODCC_API std::string emit_cpp_source(const Module& module_, const printer_options& opt) {
    auto name           = module_.module_name();
    auto namespace_name = "kernel_" + name;
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

    io::pfxstringstream out;

    ENTER(out);
    out <<
        "#include <algorithm>\n"
        "#include <cmath>\n"
        "#include <cstddef>\n"
        "#include <memory>\n"
        "#include <"  << arb_header_prefix() << "mechanism_abi.h>\n"
        "#include <" << arb_header_prefix() << "math.hpp>\n";

    if (with_simd) {
        out << "#include <" << arb_header_prefix() << "simd/simd.hpp>\n";
        out << "#undef NDEBUG\n";
        out << "#include <cassert>\n";
    }

    out <<"\n"
        << namespace_declaration_open(ns_components)
        << "namespace " << namespace_name << " {\n"
        << "\n"
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
        "using ::std::sqrt;\n"
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
            out << "S::simd_abi::native_width<arb_value_type>::value;\n";
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
            "using simd_value = S::simd<arb_value_type, vector_length_, " << abi << ">;\n"
            "using simd_index = S::simd<arb_index_type, vector_length_, " << abi << ">;\n"
            "using simd_mask  = S::simd_mask<arb_value_type, vector_length_, "<< abi << ">;\n"
            "static constexpr unsigned min_align_ = std::max(S::min_align(simd_value{}), S::min_align(simd_index{}));\n"
            "\n"
            "inline simd_value safeinv(simd_value x) {\n"
            "    simd_value ones = simd_cast<simd_value>(1.0);\n"
            "    auto mask = S::cmp_eq(S::add(x,ones), ones);\n"
            "    S::where(mask, x) = simd_cast<simd_value>(DBL_EPSILON);\n"
            "    return S::div(ones, x);\n"
            "}\n"
            "\n";
    } else {
       out << "static constexpr unsigned simd_width_ = 1;\n"
              "static constexpr unsigned min_align_ = std::max(alignof(arb_value_type), alignof(arb_index_type));\n\n";
    }

    // Make implementations
    auto emit_body = [&](APIMethod *p, bool add=false) {
        auto flags = ApiFlags{}
            .additive(add)
            .point(moduleKind::point == module_.kind())
            .voltage(moduleKind::voltage == module_.kind());
        if (with_simd) {
            emit_simd_api_body(out, p, vars.scalars, flags);
        } else {
            emit_api_body(out, p, flags);
        }
    };

    const auto& [state_ids, global_ids, param_ids, white_noise_ids] = public_variable_ids(module_);
    const auto& assigned_ids = module_.assigned_block().parameters;
    out << fmt::format(FMT_COMPILE("#define PPACK_IFACE_BLOCK \\\n"
                                   "[[maybe_unused]] auto  {0}width             = pp->width;\\\n"
                                   "[[maybe_unused]] auto  {0}n_detectors       = pp->n_detectors;\\\n"
                                   "[[maybe_unused]] auto* {0}vec_ci            = pp->vec_ci;\\\n"
                                   "[[maybe_unused]] auto* {0}vec_di            = pp->vec_di;\\\n"
                                   "[[maybe_unused]] auto* {0}vec_dt            = pp->vec_dt;\\\n"
                                   "[[maybe_unused]] auto* {0}vec_v             = pp->vec_v;\\\n"
                                   "[[maybe_unused]] auto* {0}vec_i             = pp->vec_i;\\\n"
                                   "[[maybe_unused]] auto* {0}vec_g             = pp->vec_g;\\\n"
                                   "[[maybe_unused]] auto* {0}temperature_degC  = pp->temperature_degC;\\\n"
                                   "[[maybe_unused]] auto* {0}diam_um           = pp->diam_um;\\\n"
                                   "[[maybe_unused]] auto* {0}time_since_spike  = pp->time_since_spike;\\\n"
                                   "[[maybe_unused]] auto* {0}node_index        = pp->node_index;\\\n"
                                   "[[maybe_unused]] auto* {0}peer_index        = pp->peer_index;\\\n"
                                   "[[maybe_unused]] auto* {0}multiplicity      = pp->multiplicity;\\\n"
                                   "[[maybe_unused]] auto* {0}weight            = pp->weight;\\\n"
                                   "[[maybe_unused]] auto& {0}events            = pp->events;\\\n"
                                   "[[maybe_unused]] auto& {0}mechanism_id      = pp->mechanism_id;\\\n"
                                   "[[maybe_unused]] auto& {0}index_constraints = pp->index_constraints;\\\n"),
                       pp_var_pfx);
    auto global = 0;
    for (const auto& scalar: global_ids) {
        out << fmt::format("[[maybe_unused]] auto {}{} = pp->globals[{}];\\\n", pp_var_pfx, scalar.name(), global);
        global++;
    }
    out << fmt::format("[[maybe_unused]] auto const * const * {}random_numbers = pp->random_numbers;\\\n", pp_var_pfx);
    auto param = 0, state = 0;
    for (const auto& array: state_ids) {
        out << fmt::format("[[maybe_unused]] auto* {}{} = pp->state_vars[{}];\\\n", pp_var_pfx, array.name(), state);
        state++;
    }
    for (const auto& array: assigned_ids) {
        out << fmt::format("[[maybe_unused]] auto* {}{} = pp->state_vars[{}];\\\n", pp_var_pfx, array.name(), state);
        state++;
    }
    for (const auto& array: param_ids) {
        out << fmt::format("[[maybe_unused]] auto* {}{} = pp->parameters[{}];\\\n", pp_var_pfx, array.name(), param);
        param++;
    }
    auto idx = 0;
    for (const auto& ion: module_.ion_deps()) {
        out << fmt::format("[[maybe_unused]] auto& {}{} = pp->ion_states[{}];\\\n",       pp_var_pfx, ion_field(ion), idx);
        out << fmt::format("[[maybe_unused]] auto* {}{} = pp->ion_states[{}].index;\\\n", pp_var_pfx, ion_index(ion), idx);
        idx++;
    }
    out << "//End of IFACEBLOCK\n\n"
        << "\n"
        << "// interface methods\n"
        << "static void init(arb_mechanism_ppack* pp) {\n" << indent;
    emit_body(init_api);
    if (init_api && init_api->body() && !init_api->body()->statements().empty()) {
        auto n = std::count_if(vars.arrays.begin(), vars.arrays.end(),
                               [] (const auto& v) { return v->is_state(); });
        out << fmt::format(FMT_COMPILE("if (!{0}multiplicity) return;\n"
                                       "for (arb_size_type ix = 0; ix < {1}; ++ix) {{\n"
                                       "    for (arb_size_type iy = 0; iy < {0}width; ++iy) {{\n"
                                       "        pp->state_vars[ix][iy] *= {0}multiplicity[iy];\n"
                                       "    }}\n"
                                       "}}\n"),
                           pp_var_pfx,
                           n);
    }
    out << popindent << "}\n\n";

    out << "static void advance_state(arb_mechanism_ppack* pp) {\n" << indent;
    emit_body(state_api);
    out << popindent << "}\n\n";

    out << "static void compute_currents(arb_mechanism_ppack* pp) {\n" << indent;
    emit_body(current_api, true);
    out << popindent << "}\n\n";

    out << "static void write_ions(arb_mechanism_ppack* pp) {\n" << indent;
    emit_body(write_ions_api);
    out << popindent << "}\n\n";

    if (net_receive_api) {
        out << fmt::format(FMT_COMPILE("static void apply_events(arb_mechanism_ppack* pp, arb_deliverable_event_stream* stream_ptr) {{\n"
                                       "    PPACK_IFACE_BLOCK;\n"
                                       "    auto ncell = stream_ptr->n_streams;\n"
                                       "    for (arb_size_type c = 0; c<ncell; ++c) {{\n"
                                       "        auto begin  = stream_ptr->events + stream_ptr->begin[c];\n"
                                       "        auto end    = stream_ptr->events + stream_ptr->end[c];\n"
                                       "        for (auto p = begin; p<end; ++p) {{\n"
                                       "            auto i_     = p->mech_index;\n"
                                       "            auto {1} = p->weight;\n"
                                       "            if (p->mech_id=={0}mechanism_id) {{\n"),
                           pp_var_pfx,
                           net_receive_api->args().empty() ? "weight" : net_receive_api->args().front()->is_argument()->name());
        out << indent << indent << indent << indent;
        emit_api_body(out, net_receive_api, net_recv_flags);
        out << popindent << "}\n" << popindent << "}\n" << popindent << "}\n" << popindent << "}\n\n";
    } else {
        out << "static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}\n\n";
    }

    if(post_event_api) {
        const std::string time_arg = post_event_api->args().empty() ? "time" : post_event_api->args().front()->is_argument()->name();
        out << fmt::format(FMT_COMPILE("static void post_event(arb_mechanism_ppack* pp) {{\n"
                                       "    PPACK_IFACE_BLOCK;\n"
                                       "    for (arb_size_type i_ = 0; i_ < {0}width; ++i_) {{\n"
                                       "        auto node_index_i_ = {0}node_index[i_];\n"
                                       "        auto cid_          = {0}vec_ci[node_index_i_];\n"
                                       "        auto offset_       = {0}n_detectors * cid_;\n"
                                       "        for (auto c = 0; c < {0}n_detectors; c++) {{\n"
                                       "            auto {1} = {0}time_since_spike[offset_ + c];\n"
                                       "            if ({1} >= 0) {{\n"),
                           pp_var_pfx,
                           time_arg);
        out << indent << indent << indent << indent;
        emit_api_body(out, post_event_api, post_evt_flags);
        out << popindent << "}\n" << popindent << "}\n" << popindent << "}\n" << popindent << "}\n";
    } else {
        out << "static void post_event(arb_mechanism_ppack*) {}\n";
    }

    out << popindent
        << "#undef PPACK_IFACE_BLOCK\n"
        << "} // namespace kernel_" << name
        << "\n"
        << namespace_declaration_close(ns_components)
        << "\n";

    std::stringstream ss;
    for (const auto& c: ns_components) ss << c << "::";
    ss << namespace_name << "::";

    out << fmt::format(FMT_COMPILE("extern \"C\" {{\n"
                                   "  arb_mechanism_interface* make_{0}_{1}_interface_multicore() {{\n"
                                   "    static arb_mechanism_interface result;\n"
                                   "    result.partition_width = {3}simd_width_;\n"
                                   "    result.backend = {2};\n"
                                   "    result.alignment = {3}min_align_;\n"
                                   "    result.init_mechanism = {3}init;\n"
                                   "    result.compute_currents = {3}compute_currents;\n"
                                   "    result.apply_events = {3}apply_events;\n"
                                   "    result.advance_state = {3}advance_state;\n"
                                   "    result.write_ions = {3}write_ions;\n"
                                   "    result.post_event = {3}post_event;\n"
                                   "    return &result;\n"
                                   "  }}\n"
                                   "}}\n\n"),
                       std::regex_replace(opt.cpp_namespace, std::regex{"::"}, "_"),
                       name,
                       "arb_backend_kind_cpu",
                       ss.str());

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

void CPrinter::visit(WhiteNoise* sym) {
    out_ << fmt::format("{}random_numbers[{}][i_]", pp_var_pfx, sym->index());
}

void CPrinter::visit(VariableExpression *sym) {
    out_ << fmt::format("{}{}{}", pp_var_pfx, sym->name(), sym->is_range() ? "[i_]": "");
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
            out_ << "arb_value_type ";
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

namespace {
    // Access through ppack
    std::string data_via_ppack(const indexed_variable_info& i) { return pp_var_pfx + i.data_var; }
    std::string node_index_i_name(const indexed_variable_info& i) { return i.node_index_var + "i_"; }
    std::string source_index_i_name(const index_prop& i) { return i.source_var + "i_"; }
    std::string source_var(const index_prop& i) { return pp_var_pfx + i.source_var; }

    // Convenience I/O wrapper for emitting indexed access to an external variable.

    struct deref {
        indexed_variable_info d;
        deref(indexed_variable_info d): d(d) {}

        friend std::ostream& operator<<(std::ostream& o, const deref& wrap) {
            auto index_var = wrap.d.outer_index_var();
            auto i_name = index_i_name(index_var);
            index_var = pp_var_pfx + index_var;
            return o << data_via_ppack(wrap.d) << '[' << (wrap.d.scalar() ? "0": i_name) << ']';
        }
    };
}

// Return the indices that need to be read at the beginning
// of an APIMethod in the order that they should be read
// eg:
//   node_index_ = node_index[i];
//   domain_index_ = vec_di[node_index_];
std::list<index_prop> gather_indexed_vars(const std::vector<LocalVariable*>& indexed_vars, const std::string& index) {
    std::list<index_prop> indices;
    for (auto& sym: indexed_vars) {
        auto d = decode_indexed_variable(sym->external_variable());
        if (!d.scalar()) {
            auto nested = !d.inner_index_var().empty();
            if (nested) {
                // Need to read 2 indices: outer[inner[index]]
                index_prop inner_index_prop = {d.inner_index_var(), index, d.index_var_kind};
                index_prop outer_index_prop = {d.outer_index_var(), d.inner_index_var()+"i_", d.index_var_kind};

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
                index_prop outer_index_prop = {d.outer_index_var(), index, d.index_var_kind};
                auto it = std::find(indices.begin(), indices.end(), outer_index_prop);

                // Check that the index hasn't already been added to the list
                if (it == indices.end()) {
                    indices.push_front(outer_index_prop);
                }
            }
        }
    }
    return indices;
}

void emit_api_body(std::ostream& out, APIMethod* method, const ApiFlags& flags) {
    ENTER(out);
    auto body = method->body();
    auto indexed_vars = indexed_locals(method->scope());

    std::list<index_prop> indices = gather_indexed_vars(indexed_vars, "i_");
    if (!body->statements().empty()) {
        if (flags.ppack_iface) out << "PPACK_IFACE_BLOCK;\n";
        if (flags.cv_loop) {
            out << fmt::format("for (arb_size_type i_ = 0; i_ < {}width; ++i_) {{\n",
                               pp_var_pfx)
                << indent;
        }
        for (auto index: indices) {
            out << fmt::format("auto {} = {}[{}];\n",
                               source_index_i_name(index),
                               source_var(index),
                               index.index_name);
        }

        for (auto& sym: indexed_vars) {
            auto d = decode_indexed_variable(sym->external_variable());
            auto write_voltage = sym->external_variable()->data_source() == sourceKind::voltage
                              && flags.can_write_voltage;
            out << "arb_value_type " << cprint(sym) << " = ";
            if (sym->is_read() || (sym->is_write() && d.additive) || write_voltage) {
                out << scaled(d.scale) << deref(d) << ";\n";
            }
            else {
                out << "0;\n";
            }
        }

        out << cprint(body);

        for (auto& sym: indexed_vars) {
            if (!sym->external_variable()->is_write()) continue;
            auto d = decode_indexed_variable(sym->external_variable());
            auto write_voltage = sym->external_variable()->data_source() == sourceKind::voltage
                              && flags.can_write_voltage;
            if (write_voltage) d.readonly = false;

            bool use_weight = d.always_use_weight || !flags.is_point;
            if (d.readonly) throw compiler_exception("Cannot assign to read-only external state: "+sym->to_string());
            std::string
                var,
                weight = use_weight ? pp_var_pfx + "weight[i_]" : "1.0",
                scale  = scaled(1.0/d.scale),
                name   = sym->name();
            {
                std::stringstream v; v << deref(d); var = v.str();
            }
            if (d.additive && flags.use_additive) {
                out << fmt::format("{3} -= {0};\n"
                                   "{0} = fma({1}{2}, {3}, {0});\n",
                                   var, scale, weight, name);
            }
            else if (write_voltage) {
                // SAFETY: we only ever allow *one* V-PROCESS per CV, so this is OK.
                out << fmt::format("{0} = {1};\n", var, name);
            }
            else if (d.accumulate) {
                out << fmt::format("{} = fma({}{}, {}, {});\n",
                                   var, scale, weight, name, var);
            }
            else {
                out << var << " = " << scale << name << ";\n";
            }
        }
        if (flags.cv_loop) out << popindent << "}\n";
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
        out_ << "simd_cast<simd_value>(indirect(" << pp_var_pfx << sym->name() << "+" << index << ", simd_width_))";
    }
    else {
        out_ << pp_var_pfx << sym->name();
    }
    EXITM(out_, "variable");
}

void SimdPrinter::visit(WhiteNoise* sym) {
    auto index = is_indirect_? "index_": "i_";
    out_ << fmt::format("simd_cast<simd_value>(indirect({}random_numbers[{}]+{}, simd_width_))",
        pp_var_pfx, sym->index(), index);
}

void SimdPrinter::visit(AssignmentExpression* e) {
    ENTERM(out_, "assign");
    if (!e->lhs() || !e->lhs()->is_identifier() || !e->lhs()->is_identifier()->symbol()) {
        throw compiler_exception("Expect symbol on lhs of assignment: "+e->to_string());
    }

    Symbol* lhs = e->lhs()->is_identifier()->symbol();
    std::string pfx = lhs->is_local_variable() ? "" : pp_var_pfx;
    std::string index = is_indirect_ ? "index_" : "i_";

    // lhs should not be an IndexedVariable, only a VariableExpression or LocalVariable.
    // IndexedVariables are only assigned in API calls and are handled in a special way.
    if (lhs->is_indexed_variable()) {
        throw (compiler_exception("Should not be trying to assign an IndexedVariable " + lhs->to_string(), lhs->location()));
    }
    // If lhs is a VariableExpression, it must be a range variable. Non-range variables
    // are scalars and read-only.
    if (lhs->is_variable() && lhs->is_variable()->is_range()) {
        out_ << "indirect(" << pfx << lhs->name() << "+" << index << ", simd_width_) = ";
        if (!input_mask_.empty())
            out_ << "S::where(" << input_mask_ << ", ";

        // If the rhs is a scalar identifier or a number, it needs to be cast to a vector.
        auto id = e->rhs()->is_identifier();
        auto num = e->rhs()->is_number();
        bool cast = num || (id && scalars_.count(id->name()));

        if (cast) out_ << "simd_cast<simd_value>(";
        e->rhs()->accept(this);
        if (cast) out_ << ")";

        if (!input_mask_.empty())
            out_ << ")";
    }
    else if (lhs->is_variable() && !lhs->is_variable()->is_range()) {
        throw (compiler_exception("Should not be trying to assign a non-range variable " + lhs->to_string(), lhs->location()));
    }
    // Otherwise, lhs must be a LocalVariable, we don't need to mask assignment according to the
    // input_mask_.
    else {
        out_ << "assign(" << pfx << lhs->name() << ", ";
        if (auto rhs = e->rhs()->is_identifier()) {
            if (auto sym = rhs->symbol()) {
                // We shouldn't call the rhs visitor in this case because it automatically casts indirect expressions
                if (sym->is_variable() && sym->is_variable()->is_range()) {
                    out_ << "indirect(" << pp_var_pfx << rhs->name() << "+" << index << ", simd_width_))";
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

void emit_simd_state_read(std::ostream& out, LocalVariable* local, simd_expr_constraint constraint, const ApiFlags& flags) {
    ENTER(out);
    out << "simd_value " << local->name();

    auto write_voltage = local->external_variable()->data_source() == sourceKind::voltage
                      && flags.can_write_voltage;
    auto is_additive = local->is_write() && decode_indexed_variable(local->external_variable()).additive;

    if (local->is_read() || is_additive || write_voltage) {
        auto d = decode_indexed_variable(local->external_variable());
        if (d.scalar()) {
            out << " = simd_cast<simd_value>(" << pp_var_pfx << d.data_var
                << "[0]);\n";
        }
        else {
            switch (d.index_var_kind) {
                case index_kind::node: {
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
                    break;
                }
                default: {
                    out << ";\n"
                        << "assign(" << local->name() << ", indirect(" << data_via_ppack(d)
                        << ", " << index_i_name(d.outer_index_var()) << ", simd_width_, index_constraint::none));\n";
                    break;
                }
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

void emit_simd_state_update(std::ostream& out,
                            Symbol* from, IndexedVariable* external,
                            simd_expr_constraint constraint,
                            const ApiFlags& flags) {
    if (!external->is_write()) return;
    ENTER(out);
    auto d = decode_indexed_variable(external);

    if (external->data_source() == sourceKind::voltage && flags.can_write_voltage) {
        d.readonly = false;
    }

    if (d.readonly) {
        throw compiler_exception("Cannot assign to read-only external state: "+external->to_string());
    }

    auto ext = external->name();
    auto name = from->name();
    auto data = data_via_ppack(d);
    auto node = node_index_i_name(d);
    auto index = index_i_name(d.outer_index_var());

    std::string scaled = name;
    if (d.scale != 1.0) {
        std::stringstream ss;
        ss << "S::mul(" << name << ", simd_cast<simd_value>(" << as_c_double(1/d.scale) << "))";
        scaled = ss.str();
    }
    auto write_voltage = external->data_source() == sourceKind::voltage
                      && flags.can_write_voltage;
    if (write_voltage) d.readonly = false;

    std::string weight = (d.always_use_weight || !flags.is_point) ? "w_" : "simd_cast<simd_value>(1.0)";

    if (d.additive && flags.use_additive) {
        if (d.index_var_kind == index_kind::node) {
            if (constraint == simd_expr_constraint::contiguous) {
                out << fmt::format("indirect({} + {}, simd_width_) = S::mul({}, {});\n",
                                   data, node, weight, scaled);
            }
            else {
                    // We need this instead of simple assignment!
                    out << fmt::format("{{\n"
                                       "  simd_value t_{}0_ = simd_cast<simd_value>(0.0);\n"
                                       "  assign(t_{}0_, indirect({}, simd_cast<simd_index>({}), simd_width_, constraint_category_));\n"
                                       "  {} -= t_{}0_;\n"
                                       "  indirect({}, simd_cast<simd_index>({}), simd_width_, constraint_category_) += S::mul({}, {});\n"
                                       "}}\n",
                                       name,
                                       name, data, node,
                                       scaled, name,
                                       data, node, weight, scaled);
            }
        }
        else {
            out << fmt::format("indirect({}, {}, simd_width_, index_constraint::none) = S::mul({}, {});\n",
                               data, index, weight, scaled);
        }
    }
    else if (write_voltage) {
        /* For voltage processes we *assign* to the potential field.
        ** SAFETY: only one V-PROCESS per CV allowed
        */
        if (d.index_var_kind == index_kind::node) {
            if (constraint == simd_expr_constraint::contiguous) {
                out << fmt::format("indirect({} + {}, simd_width_) = {};\n",
                                   data, node, name);
            }
            else {
                // We need this instead of simple assignment!
                out << fmt::format("{{\n"
                                   "  simd_value t_{}0_ = simd_cast<simd_value>(0.0);\n"
                                   "  assign(t_{}0_, indirect({}, simd_cast<simd_index>({}), simd_width_, constraint_category_));\n"
                                   "  {} -= t_{}0_;\n"
                                   "  indirect({}, simd_cast<simd_index>({}), simd_width_, constraint_category_) += S::mul({}, {});\n"
                                   "}}\n",
                                   name,
                                   name, data, node,
                                   scaled, name,
                                   data, node, weight, scaled);
            }
        }
        else {
            out << fmt::format("indirect({}, {}, simd_width_, index_constraint::none) = {};\n",
                               data, index, name);
        }
    }
    else if (d.accumulate) {
        if (d.index_var_kind == index_kind::node) {
            std::string tempvar = "t_" + external->name();
            switch (constraint) {
                case simd_expr_constraint::contiguous:
                    out << "simd_value " << tempvar << ";\n"
                        << "assign(" << tempvar << ", indirect(" << data << " + " << node << ", simd_width_));\n"
                        << tempvar << " = S::fma(" << weight << ", " << scaled << ", " << tempvar << ");\n"
                        << "indirect(" << data << " + " << node << ", simd_width_) = " << tempvar << ";\n";
                    break;
                case simd_expr_constraint::constant:
                    out << "indirect(" << data << ", simd_cast<simd_index>(" << node << "), simd_width_, constraint_category_) += S::mul(" << weight << ", " << scaled << ");\n";
                    break;
                default:
                    out << "indirect(" << data << ", " << node << ", simd_width_, constraint_category_) += S::mul(" << weight << ", " << scaled << ");\n";
            }
        } else {
            out << "indirect(" << data << ", " << index << ", simd_width_, index_constraint::none) += S::mul(" << weight << ", " << scaled << ");\n";
        }
    }
    else if (d.index_var_kind == index_kind::node) {
        switch (constraint) {
            case simd_expr_constraint::contiguous:
                out << "indirect(" << data << " + " << node << ", simd_width_) = " << scaled << ";\n";
                break;
            case simd_expr_constraint::constant:
                out << "indirect(" << data << ", simd_cast<simd_index>(" << node << "), simd_width_, constraint_category_) = " << scaled << ";\n";
                break;
            default:
                out << "indirect(" << data << ", " << node << ", simd_width_, constraint_category_) = " << scaled << ";\n";
        }
    }
    else {
        out << "indirect(" << data << ", " << index << ", simd_width_, index_constraint::none) = " << scaled << ";\n";
    }
    EXIT(out);
}

void emit_simd_index_initialize(std::ostream& out, const std::list<index_prop>& indices,
                                simd_expr_constraint constraint) {
    ENTER(out);
    for (auto& index: indices) {
        switch (index.kind) {
            case index_kind::node: {
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
                break;
            }
            case index_kind::cell: {
                // Treat like reading a state variable.
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
                break;
            }
            default: {
                out << "auto " << source_index_i_name(index) << " = simd_cast<simd_index>(indirect(&" << source_var(index)
                    << "[0] + " << index.index_name << ", simd_width_));\n";
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
        const simd_expr_constraint& constraint,
        const ApiFlags& flags) {
    ENTER(out);
    emit_simd_index_initialize(out, indices, constraint);

    for (auto& sym: indexed_vars) {
        emit_simd_state_read(out, sym, constraint, flags);
    }

    simdprint printer(body, scalars);
    printer.set_indirect_index();

    out << printer;

    for (auto& sym: indexed_vars) {
        emit_simd_state_update(out, sym, sym->external_variable(), constraint, flags);
    }
    EXIT(out);
}

void emit_simd_for_loop_per_constraint(std::ostream& out, BlockExpression* body,
                                       const std::vector<LocalVariable*>& indexed_vars,
                                       const std::vector<VariableExpression*>& scalars,
                                       const std::list<index_prop>& indices,
                                       const simd_expr_constraint& constraint,
                                       std::string underlying_constraint_name,
                                       const ApiFlags& flags) {
    ENTER(out);
    out << fmt::format("constraint_category_ = index_constraint::{1};\n"
                       "for (auto i_ = 0ul; i_ < {0}index_constraints.n_{1}; i_++) {{\n"
                       "    arb_index_type index_ = {0}index_constraints.{1}[i_];\n",
                       pp_var_pfx,
                       underlying_constraint_name)
        << indent
        << fmt::format("simd_value w_;\n"
                       "assign(w_, indirect(({}weight+index_), simd_width_));\n",
                       pp_var_pfx);

    emit_simd_body_for_loop(out, body, indexed_vars, scalars, indices, constraint, flags);

    out << popindent << "}\n";
    EXIT(out);
}

void emit_simd_api_body(std::ostream& out, APIMethod* method,
                        const std::vector<VariableExpression*>& scalars,
                        const ApiFlags& flags) {
    auto body = method->body();
    auto indexed_vars = indexed_locals(method->scope());

    ENTER(out);
    std::list<index_prop> indices = gather_indexed_vars(indexed_vars, "index_");
    std::vector<LocalVariable*> scalar_indexed_vars;
    for (auto& sym: indexed_vars) {
        if (decode_indexed_variable(sym->external_variable()).scalar()) {
            scalar_indexed_vars.push_back(sym);
        }
    }
    if (!body->statements().empty()) {
        out << "PPACK_IFACE_BLOCK;\n";
        out << "assert(simd_width_ <= (unsigned)S::width(simd_cast<simd_value>(0)));\n";
        if (!indices.empty()) {
            out << "index_constraint constraint_category_;\n\n";

            //Generate for loop for all contiguous simd_vectors
            simd_expr_constraint constraint = simd_expr_constraint::contiguous;
            std::string underlying_constraint = "contiguous";

            emit_simd_for_loop_per_constraint(out, body, indexed_vars, scalars, indices, constraint, underlying_constraint, flags);

            //Generate for loop for all independent simd_vectors
            constraint = simd_expr_constraint::other;
            underlying_constraint = "independent";

            emit_simd_for_loop_per_constraint(out, body, indexed_vars, scalars, indices, constraint, underlying_constraint, flags);

            //Generate for loop for all simd_vectors that have no optimizing constraints
            constraint = simd_expr_constraint::other;
            underlying_constraint = "none";

            emit_simd_for_loop_per_constraint(out, body, indexed_vars, scalars, indices, constraint, underlying_constraint, flags);

            //Generate for loop for all constant simd_vectors
            constraint = simd_expr_constraint::constant;
            underlying_constraint = "constant";

            emit_simd_for_loop_per_constraint(out, body, indexed_vars, scalars, indices, constraint, underlying_constraint, flags);
        }
        else {
            // We may nonetheless need to read a global scalar indexed variable.
            for (auto& sym: scalar_indexed_vars) {
                emit_simd_state_read(out, sym, simd_expr_constraint::other, flags);
            }

            out << fmt::format("for (arb_size_type i_ = 0; i_ < {}width; i_ += simd_width_) {{\n",
                               pp_var_pfx)
                << indent
                << simdprint(body, scalars)
                << popindent
                <<
                "}\n";
        }
    }
    EXIT(out);
}
