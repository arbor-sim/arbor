#pragma once

#include <iostream>
#include "modccutil.hpp"

enum class targetKind {
    cpu,
    gpu,
    // Vectorisation targets
    avx2,
    avx512
 };

struct Options {
    std::string filename;
    std::string outputname;
    std::string modulename;
    bool has_output = false;
    bool verbose = true;
    bool optimize = false;
    bool analysis = false;
    targetKind target = targetKind::cpu;

    void print() {
        std::cout << cyan("." + std::string(60, '-') + ".") << "\n";
        std::cout << cyan("| file     ") << filename
                  << std::string(61-11-filename.size(),' ')
                  << cyan("|") << "\n";

        std::string outname = (outputname.size() ? outputname : "stdout");
        std::cout << cyan("| output   ") << outname
                  << std::string(61-11-outname.size(),' ')
                  << cyan("|") << "\n";
        std::cout << cyan("| verbose  ") << (verbose  ? "yes" : "no ")
                  << std::string(61-11-3,' ') << cyan("|") << "\n";
        std::cout << cyan("| optimize ") << (optimize ? "yes" : "no ")
                  << std::string(61-11-3,' ') << cyan("|") << "\n";
        std::cout << cyan("| target   ")
                  << (target==targetKind::cpu? "cpu" : "gpu")
                  << std::string(61-11-3,' ') << cyan("|") << "\n";
        std::cout << cyan("| analysis ") << (analysis ? "yes" : "no ")
                  << std::string(61-11-3,' ') << cyan("|") << "\n";
        std::cout << cyan("." + std::string(60, '-') + ".") << std::endl;
    }

    Options(const Options& other) = delete;
    void operator=(const Options& other) = delete;

    static Options& instance() {
        static Options instance;
        return instance;
    }

private:
    Options() {}
};
