#include <ostream>
#include <set>
#include <string>

#include "blocks.hpp"
#include "infoprinter.hpp"
#include "module.hpp"
#include "printerutil.hpp"

#include "io/ostream_wrappers.hpp"
#include "io/prefixbuf.hpp"

using io::quote;

struct id_field_info {
    id_field_info(const Id& id, const char* kind): id(id), kind(kind) {}

    const Id& id;
    const char* kind;
};

std::ostream& operator<<(std::ostream& out, const id_field_info& wrap) {
    const Id& id = wrap.id;

    out << "{" << quote(id.name()) << ", "
        << "{spec::" << wrap.kind << ", " << quote(id.unit_string()) << ", "
        << (id.has_value()? id.value: "0");

    if (id.has_range()) {
        out << ", " << id.range.first.spelling << "," << id.range.second.spelling;
    }

    out << "}}";
    return out;
}

struct ion_dep_info {
    ion_dep_info(const IonDep& ion): ion(ion) {}

    const IonDep& ion;
};

std::ostream& operator<<(std::ostream& out, const ion_dep_info& wrap) {
    const char* boolalpha[2] = {"false", "true"};
    const IonDep& ion = wrap.ion;

    return out << "{\"" << ion.name << "\", {"
        << boolalpha[ion.writes_concentration_int()] << ", "
        << boolalpha[ion.writes_concentration_ext()] << ", "
        << boolalpha[ion.uses_rev_potential()] << ", "
        << boolalpha[ion.writes_rev_potential()] << ", "
        << boolalpha[ion.uses_valence()] << ", "
        << boolalpha[ion.verifies_valence()] << ", "
        << ion.expected_valence << "}}";
}

std::string build_info_header(const Module& m, const printer_options& opt) {
    using io::indent;
    using io::popindent;

    std::string name = m.module_name();
    auto ids = public_variable_ids(m);
    auto ns_components = namespace_components(opt.cpp_namespace);

    bool any_fields =
        !ids.global_parameter_ids.empty() ||
        !ids.range_parameter_ids.empty() ||
        !ids.state_ids.empty();

    io::pfxstringstream out;

    out <<
        "#pragma once\n"
        "#include <memory>\n"
        "\n"
        "#include <" << arb_header_prefix() << "mechanism.hpp>\n"
        "#include <" << arb_header_prefix() << "mechinfo.hpp>\n"
        "\n"
        << namespace_declaration_open(ns_components) <<
        "\n"
        "template <typename Backend>\n"
        "::arb::concrete_mech_ptr<Backend> make_mechanism_" << name << "();\n"
        "\n"
        "inline const ::arb::mechanism_info& mechanism_" << name << "_info() {\n"
        << indent;

   any_fields && out <<
        "using spec = ::arb::mechanism_field_spec;\n";

   out <<
        "static ::arb::mechanism_info info = {\n"
        << indent <<
        "// globals\n"
        "{\n"
        << indent;

    io::separator sep(",\n");
    for (const auto& id: ids.global_parameter_ids) {
        out << sep << id_field_info(id, "global");
    }

    out << popindent <<
        "\n},\n// parameters\n{\n"
        << indent;

    sep.reset();
    for (const auto& id: ids.range_parameter_ids) {
        out << sep << id_field_info(id, "parameter");
    }

    out << popindent <<
        "\n},\n// state variables\n{\n"
        << indent;

    sep.reset();
    for (const auto& id: ids.state_ids) {
        out << sep << id_field_info(id, "state");
    }

    out << popindent <<
        "\n},\n// ion dependencies\n{\n"
        << indent;

    sep.reset();
    for (const auto& ion: m.ion_deps()) {
        out << sep << ion_dep_info(ion);
    }

    std::string fingerprint = "<placeholder>";
    out << popindent << "\n"
        "},\n"
        "// fingerprint\n" << quote(fingerprint) << ",\n"
        "// linear, homogeneous mechanism\n" << m.is_linear() << "\n"
        << popindent <<
        "};\n"
        "\n"
        "return info;\n"
        << popindent <<
        "}\n"
        "\n"
        << namespace_declaration_close(ns_components);

    return out.str();
}
