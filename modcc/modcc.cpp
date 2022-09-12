#include <exception>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <regex>

#include <tinyopt/tinyopt.h>

#include "printer/cprinter.hpp"
#include "printer/gpuprinter.hpp"
#include "printer/infoprinter.hpp"
#include "printer/printeropt.hpp"
#include "printer/simd.hpp"

#include "module.hpp"
#include "parser.hpp"
#include "perfvisitor.hpp"

#include "io/bulkio.hpp"
#include "io/pprintf.hpp"

#include <fmt/format.h>

using std::cout;
using std::cerr;

// Options and option parsing:

int report_error(const std::string& message) {
    cerr << red("error trace:\n") << message << "\n";
    return 1;
}

int report_ice(const std::string& message) {
    cerr << red("internal compiler error:\n") << message << "\n"
         << "\nPlease report this error to the modcc developers.\n";
    return 1;
}

enum class targetKind {
    cpu,
    gpu,
};

std::unordered_map<std::string, targetKind> targetKindMap = {
    {"cpu", targetKind::cpu},
    {"gpu", targetKind::gpu},
};

std::unordered_map<std::string, enum simd_spec::simd_abi> simdAbiMap = {
    {"none", simd_spec::none},
    {"neon", simd_spec::neon},
    {"sve", simd_spec::sve},
    {"avx",  simd_spec::avx},
    {"avx2", simd_spec::avx2},
    {"avx512", simd_spec::avx512},
    {"default_abi", simd_spec::default_abi},
    {"native", simd_spec::native}
};

template <typename Map, typename V>
auto key_by_value(const Map& map, const V& v) -> decltype(map.begin()->first) {
    for (const auto& kv: map) {
        if (kv.second==v) return kv.first;
    }
    throw std::out_of_range("value not found in map");
}

struct Options {
    std::string outprefix;
    std::vector<std::string> modfiles;
    std::string modulename;
    std::string catalogue;
    bool verbose = false;
    bool analysis = false;
    std::unordered_set<targetKind> targets;
};

// Helper for formatting tabulated output (option reporting).
struct table_prefix { std::string text; };
std::ostream& operator<<(std::ostream& out, const table_prefix& tb) {
    return out << cyan("| "+tb.text) << std::right << std::setw(58-tb.text.size());
};

std::ostream& operator<<(std::ostream& out, simd_spec simd) {
    std::stringstream s;
    s << key_by_value(simdAbiMap, simd.abi);
    if (simd.width!=0) {
        s << '/' << simd.width;
    }
    return out << s.str();
}

std::ostream& operator<<(std::ostream& out, const Options& opt) {
    static const char* noyes[2] = {"no", "yes"};
    static const std::string line_end = cyan(" |") + "\n";

    std::string targets;
    for (targetKind t: opt.targets) {
        targets += " "+key_by_value(targetKindMap, t);
    }

    for (const auto& f: opt.modfiles) {
        out << table_prefix{"file"} << f << line_end;
    }
    out << table_prefix{"output"} << (opt.outprefix.empty()? "-": opt.outprefix) << line_end <<
        table_prefix{"verbose"} << noyes[opt.verbose] << line_end <<
        table_prefix{"targets"} << targets << line_end <<
        table_prefix{"analysis"} << noyes[opt.analysis] << line_end;
    return out;
}

std::ostream& operator<<(std::ostream& out, const printer_options& popt) {
    static const std::string line_end = cyan(" |") + "\n";

    return out <<
        table_prefix{"namespace"} << popt.cpp_namespace << line_end <<
        table_prefix{"simd"} << popt.simd << line_end;
}

std::istream& operator>> (std::istream& i, simd_spec& spec) {
    auto npos = std::string::npos;
    std::string s;
    i >> s;
    unsigned width = no_size;

    auto suffix = s.find_last_of('/');
    if (suffix!=npos) {
        width = stoul(s.substr(suffix+1));
        s = s.substr(0, suffix);
    }

    spec = simd_spec(simdAbiMap.at(s.c_str()), width);
    return i;
}

const char* usage_str =
        "\n"
        "-o|--output            [Prefix for output file names]\n"
        "-N|--namespace         [Namespace for generated code]\n"
        "-t|--target            [Build module for target; Avaliable targets: 'cpu', 'gpu']\n"
        "-s|--simd              [Generate code with explicit SIMD vectorization]\n"
        "-S|--simd-abi          [Override SIMD ABI in generated code. Use /n suffix to force SIMD width to be size n. Examples: 'avx2', 'native/4', ...]\n"
        "-V|--verbose           [Toggle verbose mode]\n"
        "-A|--analyse           [Toggle analysis mode]\n"
        "-T|--trace-codegen     [Leave trace marks in generated source]\n"
        "<filenames>            [Files to be compiled]\n";

int main(int argc, char **argv) {
    using namespace to;

    Options opt;
    printer_options popt;
    try {
        std::vector<std::string> targets;

        auto help = [argv0 = argv[0]] {
            to::usage(argv0, usage_str);
        };

        auto enable_simd = [&popt] {
            if (popt.simd.abi==simd_spec::none) {
                popt.simd = simd_spec(simd_spec::native);
            }
        };

        auto add_target = [&opt](targetKind t) { opt.targets.insert(t); };

        to::option options[] = {
                { to::push_back(opt.modfiles)},
                { opt.outprefix,                                         "-o", "--output-dir" },
                { to::set(opt.verbose),  to::flag,                       "-V", "--verbose" },
                { to::set(opt.analysis), to::flag,                       "-A", "--analyse" },
                { opt.catalogue,                                         "-c", "--catalogue"},
                { popt.cpp_namespace,                                    "-N", "--namespace" },
                { to::action(enable_simd), to::flag,                     "-s", "--simd" },
                { popt.simd,                                             "-S", "--simd-abi" },
                { to::set(popt.trace_codegen), to::flag,                 "-T", "--trace-codegen"},
                { {to::action(add_target, to::keywords(targetKindMap))}, "-t", "--target" },
                { to::action(help), to::flag, to::exit,                  "-h", "--help" }
        };

        if (!to::run(options, argc, argv+1)) return 0;
    }
    catch (to::option_error& e) {
        to::usage_error(argv[0], usage_str, e.what());
        return 1;
    }

    if (!opt.catalogue.empty()) popt.cpp_namespace += "::" + opt.catalogue + "_catalogue";

    std::vector<std::string> modules;

    for (const auto& modfile: opt.modfiles) {
        try {
            auto emit_header = [&opt](const char* h) {
                if (opt.verbose) {
                    cout << green("[") << h << green("]") << "\n";
                }
            };

            if (opt.verbose) {
                static const std::string tableline = cyan("."+std::string(60, '-')+".")+"\n";
                cout << tableline;
                cout << opt;
                cout << popt;
                cout << tableline;
            }

            // Load module file and initialize Module object.
            Module m(io::read_all(modfile), modfile);

            if (m.empty()) {
                return report_error("empty file: "+modfile);
            }

            // Perform parsing and semantic analysis passes.

            emit_header("parsing");
            Parser p(m, false);
            if (!p.parse()) {
                // Parser::parse() writes its own errors to stderr.
                return 1;
            }

            emit_header("semantic analysis");
            m.semantic();
            if (m.has_warning()) {
                cerr << yellow("Warnings:\n");
                cerr << m.warning_string() << "\n";
            }
            if (m.has_error()) {
                return report_error(m.error_string());
            }

            // Generate backend-specific sources for each backend provided.

            emit_header("code generation");

            std::string prefix = m.module_name();
            if (!opt.outprefix.empty()) {
                if (opt.outprefix.back() != '/') opt.outprefix += "/";
                prefix = opt.outprefix + prefix;
            }

            bool have_cpu = opt.targets.find(targetKind::cpu) != opt.targets.end();
            bool have_gpu = opt.targets.find(targetKind::gpu) != opt.targets.end();

            io::write_all(build_info_header(m, popt, have_cpu, have_gpu), prefix+".hpp");
            for (targetKind target: opt.targets) {
                std::string outfile = prefix;
                switch (target) {
                    case targetKind::gpu:
                        io::write_all(emit_gpu_cpp_source(m, popt), outfile+"_gpu.cpp");
                        io::write_all(emit_gpu_cu_source(m, popt), outfile+"_gpu.cu");
                        break;
                    case targetKind::cpu:
                        io::write_all(emit_cpp_source(m, popt), outfile+"_cpu.cpp");
                        break;
                }
            }

            // Optional analysis report.

            if (opt.analysis) {
                cout << green("performance analysis\n");
                for (auto &symbol: m.symbols()) {
                    if (auto method = symbol.second->is_api_method()) {
                        cout << white("-------------------------\n");
                        cout << yellow("method " + method->name()) << "\n";
                        cout << white("-------------------------\n");

                        FlopVisitor flops;
                        method->accept(&flops);
                        cout << white("FLOPS\n") << flops.print() << "\n";

                        MemOpVisitor memops;
                        method->accept(&memops);
                        cout << white("MEMOPS\n") << memops.print() << "\n";
                    }
                }
            }

            modules.push_back(m.module_name());
        }
        catch (io::bulkio_error& e) {
            return report_error(e.what());
        }
        catch (compiler_exception& e) {
            return report_ice(pprintf("% @ %", e.what(), e.location()));
        }
        catch (std::exception& e) {
            return report_ice(e.what());
        }
        catch (...) {
            return report_ice("");
        }
    }

    if (!opt.catalogue.empty()) {
        const auto prefix = std::regex_replace(popt.cpp_namespace, std::regex{"::"}, "_");
        {
            std::ofstream out(opt.outprefix + opt.catalogue + "_catalogue.cpp");
            out << "// Automatically generated by modcc\n"
                "\n"
                "#include <arbor/mechanism_abi.h>\n"
                "\n";

            for (const auto& mod: modules) {
                out << fmt::format("#include \"{}.hpp\"\n", mod);
            }

            out << "\n"
                "#ifdef STANDALONE\n"
                "extern \"C\" {\n"
                "    [[gnu::visibility(\"default\")]] const void* get_catalogue(int* n) {\n";
            out << fmt::format("        *n = {0};\n"
                               "        static arb_mechanism cat[{0}] = {{\n",
                               opt.modfiles.size());
            for (const auto& mod: modules) {
                out << fmt::format("            make_{}_{}(),\n", prefix, mod);
            }
            out << "        };\n"
                "        return (void*)cat;\n"
                "    }\n"
                "}\n"
                "\n"
                "#else\n"
                "\n"
                "#include <arbor/mechanism.hpp>\n"
                "#include <arbor/assert.hpp>\n"
                "\n";
            out << fmt::format("#include \"{0}_catalogue.hpp\"\n"
                               "\n"
                               "namespace arb {{\n"
                               "mechanism_catalogue build_{0}_catalogue() {{\n"
                               "    mechanism_catalogue cat;\n",
                               opt.catalogue);
            for (const auto& mod: modules) {
                out << fmt::format("    {{\n"
                                   "        auto mech = make_{}_{}();\n"
                                   "        auto ty = mech.type();\n"
                                   "        auto nm = ty.name;\n"
                                   "        auto ig = mech.i_gpu();\n"
                                   "        auto ic = mech.i_cpu();\n"
                                   "        arb_assert(ic || ig);\n"
                                   "        cat.add(nm, ty);\n"
                                   "        if (ic) cat.register_implementation(nm, std::make_unique<arb::mechanism>(ty, *ic));\n"
                                   "        if (ig) cat.register_implementation(nm, std::make_unique<arb::mechanism>(ty, *ig));\n"
                                   "    }}\n",
                                   prefix, mod);
            }

            out << "    return cat;\n"
                "}\n"
                "\n";
            out << fmt::format("ARB_ARBOR_API const mechanism_catalogue& global_{0}_catalogue() {{\n"
                               "    static mechanism_catalogue cat = build_{0}_catalogue();\n"
                               "    return cat;\n"
                               "}}\n",
                               opt.catalogue);
            out << "} // namespace arb\n"
                "#endif\n";
        }
        {
        std::ofstream out(opt.outprefix + opt.catalogue + "_catalogue.hpp");
        out << fmt::format("#pragma once\n"
                           "\n"
                           "#include <arbor/mechcat.hpp>\n"
                           "#include <arbor/export.hpp>\n"
                           "\n"
                           "namespace arb {{\n"
                           "ARB_ARBOR_API const mechanism_catalogue& global_{0}_catalogue();\n"
                           "}}\n",
                           opt.catalogue);
        }
    }
}
