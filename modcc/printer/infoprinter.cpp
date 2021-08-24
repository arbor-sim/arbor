#include <ostream>
#include <set>
#include <string>
#include <regex>

#define FMT_HEADER_ONLY YES
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/compile.h>

#include "blocks.hpp"
#include "infoprinter.hpp"
#include "module.hpp"
#include "printerutil.hpp"

#include "io/ostream_wrappers.hpp"
#include "io/prefixbuf.hpp"

using io::quote;

std::string build_info_header(const Module& m, const printer_options& opt, bool cpu, bool gpu) {
    using io::indent;
    using io::popindent;

    std::string name = m.module_name();

    io::pfxstringstream out;

    std::string fingerprint = "<placeholder>";

    const auto lowest = std::to_string(std::numeric_limits<double>::lowest());
    const auto max    = std::to_string(std::numeric_limits<double>::max());
    out << fmt::format("#pragma once\n\n"
                       "#include <cmath>\n"
                       "#include <{}mechanism_abi.h>\n\n",
                       arb_header_prefix());

    const auto& [state_ids, global_ids, param_ids] = public_variable_ids(m);
    const auto& assigned_ids = m.assigned_block().parameters;
    auto fmt_var = [&](const auto& id) {
        auto lo  = id.has_range() ? id.range.first  : lowest;
        auto hi  = id.has_range() ? id.range.second : max;
        auto val = id.has_value() ? id.value        : "NAN";
        return fmt::format(FMT_COMPILE("{{ \"{}\", \"{}\", {}, {}, {} }}"), id.name(), id.unit_string(), val, lo, hi);
    };

    out << fmt::format("extern \"C\" {{\n"
                       "  arb_mechanism_type make_{0}_{1}() {{\n"
                       "    // Tables\n",
                       std::regex_replace(opt.cpp_namespace, std::regex{"::"}, "_"),
                       name);
    {
        io::separator sep("", ",\n                                        ");
        out << "    static arb_field_info globals[] = { ";
        for (const auto& var: global_ids) out << sep << fmt_var(var);
        out << fmt::format(" }};\n"
                           "    static arb_size_type n_globals = {};\n", global_ids.size());
    }
    {
        io::separator sep("", ",\n                                           ");
        out << "    static arb_field_info state_vars[] = { ";
        for (const auto& id: state_ids)    out << sep << fmt_var(id);
        for (const auto& id: assigned_ids) out << sep << fmt_var(id);
        out << fmt::format(" }};\n"
                           "    static arb_size_type n_state_vars = {};\n", assigned_ids.size() + state_ids.size());
    }
    {
        io::separator sep("", ",\n                                           ");
        out << "    static arb_field_info parameters[] = { ";
        for (const auto& id: param_ids) out << sep << fmt_var(id);
        out << fmt::format(" }};\n"
                           "    static arb_size_type n_parameters = {};\n", param_ids.size());
    }
    {
        io::separator sep("", ",\n");
        out << "    static arb_ion_info ions[] = { ";
        for (const auto& ion: m.ion_deps()) out << sep
                                                << fmt::format(FMT_COMPILE("{{ \"{}\", {}, {}, {}, {}, {}, {}, {} }}"),
                                                               ion.name,
                                                               ion.writes_concentration_int(), ion.writes_concentration_ext(),
                                                               ion.writes_rev_potential(), ion.uses_rev_potential(),
                                                               ion.uses_valence(), ion.verifies_valence(), ion.expected_valence);
        out << fmt::format(" }};\n"
                           "    static arb_size_type n_ions = {};\n", m.ion_deps().size());
    }

    out << fmt::format(FMT_COMPILE("\n"
                                   "    arb_mechanism_type result;\n"
                                   "    result.abi_version=ARB_MECH_ABI_VERSION;\n"
                                   "    result.fingerprint=\"{1}\";\n"
                                   "    result.name=\"{0}\";\n"
                                   "    result.kind={2};\n"
                                   "    result.is_linear={3};\n"
                                   "    result.has_post_events={4};\n"
                                   "    result.globals=globals;\n"
                                   "    result.n_globals=n_globals;\n"
                                   "    result.ions=ions;\n"
                                   "    result.n_ions=n_ions;\n"
                                   "    result.state_vars=state_vars;\n"
                                   "    result.n_state_vars=n_state_vars;\n"
                                   "    result.parameters=parameters;\n"
                                   "    result.n_parameters=n_parameters;\n"
                                   "    return result;\n"
                                   "  }}\n"
                                   "\n"),
                       name,
                       fingerprint,
                       module_kind_str(m),
                       m.is_linear(),
                       m.has_post_events())
        << fmt::format("  arb_mechanism_interface* make_{0}_{1}_interface_multicore(){2}\n"
                       "  arb_mechanism_interface* make_{0}_{1}_interface_gpu(){3}\n"
                       "}}\n",
                       std::regex_replace(opt.cpp_namespace, std::regex{"::"}, "_"),
                       name,
                       cpu ? ";" : " { return nullptr; }",
                       gpu ? ";" : " { return nullptr; }");
    return out.str();
}
