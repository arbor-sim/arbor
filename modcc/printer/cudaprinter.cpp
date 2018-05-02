#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>

#include "expression.hpp"
#include "io/ostream_wrappers.hpp"
#include "io/prefixbuf.hpp"
#include "printer/cexpr_emit.hpp"
#include "printer/printerutil.hpp"

using io::indent;
using io::popindent;
using io::quote;

// Emit stream_stat alis, parameter pack struct.
void emit_common_defs(std::ostream&, const Module& module_);

std::string make_class_name(const std::string& module_name) {
    return "mechanism_gpu_"+module_name;
}

std::string make_ppack_name(const std::string& module_name) {
    return make_class_name(module_name)+"_pp_";
}

static std::string ion_state_field(std::string ion_name) {
    return "ion_"+ion_name+"_";
}

static std::string ion_state_index(std::string ion_name) {
    return "ion_"+ion_name+"_index_";
}

std::string emit_cuda_cpp_source(const Module& module_, const std::string& ns) {
    std::string name = module_.module_name();
    std::string class_name = make_class_name(name);
    std::string ppack_name = make_ppack_name(name);
    auto ns_components = namespace_components(ns);

    NetReceiveExpression* net_receive = find_net_receive(module_);

    auto vars = local_module_variables(module_);
    auto ion_deps = module_.ion_deps();

    std::string fingerprint = "<placeholder>";

    io::pfxstringstream out;

    net_receive && out <<
        "#include <" << arb_header_prefix() << "backends/event.hpp>\n"
        "#include <" << arb_header_prefix() << "backends/multi_event_stream_state.hpp>\n";

    out <<
        "#include <" << arb_header_prefix() << "backends/gpu/mechanism.hpp>\n"
        "#include <" << arb_header_prefix() << "backends/gpu/mechanism_ppack_base.hpp>\n"
        "\n" << namespace_declaration_open(ns_components) <<
        "\n";

    emit_common_defs(out, module_);

    out <<
        "void " << class_name << "_nrn_init_(int, " << ppack_name << "&);\n"
        "void " << class_name << "_nrn_state_(int, " << ppack_name << "&);\n"
        "void " << class_name << "_nrn_current_(int, " << ppack_name << "&);\n"
        "void " << class_name << "_write_ions_(int, " << ppack_name << "&);\n";

    net_receive && out <<
        "void " << class_name << "_deliver_events_(int, " << ppack_name << "&, deliverable_event_stream_state events);\n";

    out <<
        "\n"
        "class " << class_name << ": public ::arb::gpu::mechanism {\n"
        "public:\n" << indent <<
        "const mechanism_fingerprint& fingerprint() const override {\n" << indent <<
        "static mechanism_fingerprint hash = " << quote(fingerprint) << ";\n"
        "return hash;\n" << popindent <<
        "}\n\n"
        "std::string internal_name() const override { return " << quote(name) << "; }\n"
        "mechanismKind kind() const override { return " << module_kind_str(module_) << "; }\n"
        "mechanism_ptr clone() const override { return mechanism_ptr(new " << class_name << "()); }\n"
        "\n"
        "void nrn_init() override {\n" << indent <<
        class_name << "_nrn_init_(width_, pp_);\n" << popindent <<
        "}\n\n"
        "void nrn_state() override {\n" << indent <<
        class_name << "_nrn_state_(width_, pp_);\n" << popindent <<
        "}\n\n"
        "void nrn_current() override {\n" << indent <<
        class_name << "_nrn_current_(width_, pp_);\n" << popindent <<
        "}\n\n"
        "void write_ions() override {\n" << indent <<
        class_name << "_write_ions_(width_, pp_);\n" << popindent <<
        "}\n\n";

    net_receive && out <<
        "void deliver_events(deliverable_event_stream_state events) override {\n" << indent <<
        class_name << "_deliver_events_(width_, pp_, events);\n" << popindent <<
        "}\n\n";

    out << popindent <<
        "protected:\n" << indent <<
        "using ionKind = ::arb::ionKind;\n\n"
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

    }

    if (!ion_deps.empty()) {
        out <<
            "mechanism_ion_state_table ion_state_table() override {\n" << indent <<
            "return {" << indent;

        sep.reset();
        for (const auto& dep: ion_deps) {
            out << sep << "{ionKind::" << dep.name << ", &pp_." << ion_state_field(dep.name) << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";

        sep.reset();
        out << "mechanism_ion_index_table ion_index_table() override {\n" << indent << "return {" << indent;
        for (const auto& dep: ion_deps) {
            out << sep << "{ionKind::" << dep.name << ", &pp_." << ion_state_index(dep.name) << "}";
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

std::string emit_cuda_cu_source(const Module& module_, const std::string& ns) {
    std::string name = module_.module_name();
    std::string class_name = make_class_name(name);
    std::string ppack_name = make_ppack_name(name);
    auto ns_components = namespace_components(ns);

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
        "#include <" << arb_header_prefix() << "backends/event.hpp>\n"
        "#include <" << arb_header_prefix() << "backends/multi_event_stream_state.hpp>\n"
        "#include <" << arb_header_prefix() << "backends/gpu/mechanism_ppack_base.hpp>\n"
        "\n" << namespace_declaration_open(ns_components) <<
        "\n";

    emit_common_defs(out, module_);

    out <<
        "using value_type = ::arb::gpu::mechanism_ppack_base::value_type;\n"
        "using index_type = ::arb::gpu::mechanism_ppack_base::index_type;\n"
        "\n";

    out <<
        "void " << class_name << "_nrn_init_(int, " << ppack_name << "&) {};\n"
        "void " << class_name << "_nrn_state_(int, " << ppack_name << "&) {};\n"
        "void " << class_name << "_nrn_current_(int, " << ppack_name << "&) {};\n"
        "void " << class_name << "_write_ions_(int, " << ppack_name << "&) {};\n";

    net_receive && out <<
        "void " << class_name << "_deliver_events_(int, " << ppack_name << "&, deliverable_event_stream_state events) {}\n";

    (void)write_ions_api;

    out << namespace_declaration_close(ns_components);
    return out.str();
}

void emit_common_defs(std::ostream& out, const Module& module_) {
    std::string class_name = make_class_name(module_.module_name());
    std::string ppack_name = make_ppack_name(module_.module_name());

    auto vars = local_module_variables(module_);
    auto ion_deps = module_.ion_deps();

    find_net_receive(module_) && out <<
        "using deliverable_event_stream_state = ::arb::multi_event_stream_state<::arb::deliverable_event_data>;\n"
        "\n";

    out <<
        "struct " << ppack_name << ": ::arb::gpu::mechanism_ppack_base {\n" << indent;

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


