#include <chrono>
#include <iostream>
#include <fstream>

#include <tclap/CmdLine.h>

#include "cprinter.hpp"
#include "cudaprinter.hpp"
#include "lexer.hpp"
#include "module.hpp"
#include "parser.hpp"
#include "perfvisitor.hpp"
#include "modccutil.hpp"
#include "options.hpp"

#include "simd_printer.hpp"

using namespace nest::mc;

//#define VERBOSE

int main(int argc, char **argv) {

    // parse command line arguments
    try {
        TCLAP::CmdLine cmd("welcome to mod2c", ' ', "0.1");

        // input file name (to load multiple files we have to use UnlabeledMultiArg
        TCLAP::UnlabeledValueArg<std::string>
            fin_arg("input_file", "the name of the .mod file to compile", true, "", "filename");
        // output filename
        TCLAP::ValueArg<std::string>
            fout_arg("o","output","name of output file", false,"","filename");
        // output filename
        TCLAP::ValueArg<std::string>
            target_arg("t","target","backend target={cpu,gpu}", true,"cpu","cpu/gpu");
        // verbose mode
        TCLAP::SwitchArg verbose_arg("V","verbose","toggle verbose mode", cmd, false);
        // analysis mode
        TCLAP::SwitchArg analysis_arg("A","analyse","toggle analysis mode", cmd, false);
        // optimization mode
        TCLAP::SwitchArg opt_arg("O","optimize","turn optimizations on", cmd, false);
        // Set module name explicitly
        TCLAP::ValueArg<std::string>
            module_arg("m", "module", "module name to use", false, "", "module");

        cmd.add(fin_arg);
        cmd.add(fout_arg);
        cmd.add(target_arg);
        cmd.add(module_arg);

        cmd.parse(argc, argv);

        Options::instance().outputname = fout_arg.getValue();
        Options::instance().has_output = Options::instance().outputname.size()>0;
        Options::instance().filename = fin_arg.getValue();
        Options::instance().modulename = module_arg.getValue();
        Options::instance().verbose = verbose_arg.getValue();
        Options::instance().optimize = opt_arg.getValue();
        Options::instance().analysis = analysis_arg.getValue();
        auto targstr = target_arg.getValue();
        if(targstr == "cpu") {
            Options::instance().target = targetKind::cpu;
        }
        else if(targstr == "gpu") {
            Options::instance().target = targetKind::gpu;
        }
        else if(targstr == "avx512") {
            Options::instance().target = targetKind::avx512;
        }
        else if(targstr == "avx2") {
            Options::instance().target = targetKind::avx2;
        }
        else {
            std::cerr << red("error")
                      << " target must be one in {cpu, gpu, avx2, avx512}\n";
            return 1;
        }
    }
    // catch any exceptions in command line handling
    catch(TCLAP::ArgException &e) {
        std::cerr << "error: "   << e.error()
                  << " for arg " << e.argId() << "\n";
    }

    try {
        // load the module from file passed as first argument
        Module m(Options::instance().filename.c_str());

        // check that the module is not empty
        if(m.buffer().size()==0) {
            std::cout << red("error: ") << white(argv[1])
                      << " invalid or empty file" << std::endl;
            return 1;
        }

        if(Options::instance().verbose) {
            Options::instance().print();
        }

        ////////////////////////////////////////////////////////////
        // parsing
        ////////////////////////////////////////////////////////////
        if(Options::instance().verbose) std::cout << green("[") + "parsing" + green("]") << std::endl;

        // initialize the parser
        Parser p(m, false);

        // parse
        p.parse();
        if(p.status() == lexerStatus::error) return 1;

        ////////////////////////////////////////////////////////////
        // semantic analysis
        ////////////////////////////////////////////////////////////
        if(Options::instance().verbose)
            std::cout << green("[") + "semantic analysis" + green("]") << "\n";

        m.semantic();

        if( m.has_error() || m.has_warning() ) {
            std::cout << m.error_string() << std::endl;
        }

        if(m.has_error()) {
            return 1;
        }

        ////////////////////////////////////////////////////////////
        // optimize
        ////////////////////////////////////////////////////////////
        if(Options::instance().optimize) {
            if(Options::instance().verbose) std::cout << green("[") + "optimize" + green("]") << std::endl;
            m.optimize();
            if(m.has_error()) {
                return 1;
            }
        }

        ////////////////////////////////////////////////////////////
        // generate output
        ////////////////////////////////////////////////////////////
        if(Options::instance().verbose) {
            std::cout << green("[") + "code generation"
                      << green("]") << std::endl;
        }

        std::string text;
        switch(Options::instance().target) {
            case targetKind::cpu  :
                text = CPrinter(m, Options::instance().optimize).emit_source();
                break;
            case targetKind::gpu  :
                text = CUDAPrinter(m, Options::instance().optimize).text();
                break;
            case targetKind::avx512:
                text = SimdPrinter<targetKind::avx512>(
                    m, Options::instance().optimize).emit_source();
                break;
            case targetKind::avx2:
                text = SimdPrinter<targetKind::avx2>(
                    m, Options::instance().optimize).emit_source();
                break;
            default :
                std::cerr << red("error") << ": unknown printer" << std::endl;
                exit(1);
        }

        if(Options::instance().has_output) {
            std::ofstream fout(Options::instance().outputname);
            fout << text;
            fout.close();
        }
        else {
            std::cout << cyan("--------------------------------------\n");
            std::cout << text;
            std::cout << cyan("--------------------------------------\n");
        }

        std::cout << yellow("successfully compiled ")
                  << white(Options::instance().filename) << " -> "
                  << white(Options::instance().outputname) << "\n";

        ////////////////////////////////////////////////////////////
        // print module information
        ////////////////////////////////////////////////////////////
        if(Options::instance().analysis) {
            std::cout << green("performance analysis") << std::endl;
            for(auto &symbol : m.symbols()) {
                if(auto method = symbol.second->is_api_method()) {
                    std::cout << white("-------------------------\n");
                    std::cout << yellow("method " + method->name()) << "\n";
                    std::cout << white("-------------------------\n");

                    FlopVisitor flops;
                    method->accept(&flops);
                    std::cout << white("FLOPS") << std::endl;
                    std::cout << flops.print() << std::endl;

                    std::cout << white("MEMOPS") << std::endl;
                    MemOpVisitor memops;
                    method->accept(&memops);
                    std::cout << memops.print() << std::endl;;
                }
            }
        }
    }

    catch(compiler_exception e) {
        std::cerr << red("internal compiler error: ")
                  << white("this means a bug in the compiler,"
                           " please report to modcc developers\n")
                  << e.what() << " @ " << e.location() << "\n";
        exit(1);
    }
    catch(std::exception e) {
        std::cerr << red("internal compiler error: ")
                  << white("this means a bug in the compiler,"
                           " please report to modcc developers\n")
                  << e.what() << "\n";
        exit(1);
    }
    catch(...) {
        std::cerr << red("internal compiler error: ")
                  << white("this means a bug in the compiler,"
                           " please report to modcc developers\n");
        exit(1);
    }

    return 0;
}
