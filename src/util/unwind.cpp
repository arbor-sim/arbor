// only compile the contents of this file if linking against libunwind
#ifdef WITH_UNWIND

#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <cxxabi.h>

#include <memory/util.hpp>
#include <util/file.hpp>
#include <util/unwind.hpp>

#include <cxxabi.h>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>

namespace nest {
namespace mc {
namespace util {

static_assert(std::is_same<std::uint32_t, unw_word_t>::value,
        "assumption that libunwind unw_word_t is unit64_t is not valid");

std::string demangle(std::string s);

///  Builds a stack trace when constructed.
///  The trace can then be printed, or accessed via the stack() member function.
backtrace::backtrace() {
    unw_cursor_t cursor;
    unw_context_t context;

    // initialize cursor to current frame for local unwinding.
    unw_getcontext(&context);
    unw_init_local(&cursor, &context);

    while (unw_step(&cursor) > 0) {
        // find the stack position
        unw_word_t offset, pc;
        unw_get_reg(&cursor, UNW_REG_IP, &pc);
        if (pc == 0) {
            break;
        }

        // get the name 
        char sym[512];
        if (unw_get_proc_name(&cursor, sym, sizeof(sym), &offset) == 0) {
            frames_.push_back({std::string(sym), pc});
        } else {
            frames_.push_back({std::string("???"), pc});
        }
    }
}

std::string demangle(std::string s) {
    int status;
    char* demangled = abi::__cxa_demangle(s.c_str(), nullptr, nullptr, &status);

    // __cxa_demangle only returns a non-empty string if it is passed a valid C++
    // mangled c++ symbol (i.e. it returns an empty string for normal c symbols)
    if (status==0) {
        s = demangled;
    }
    std::free(demangled); // don't leak the demangled string

    return s;
}

void backtrace::print(bool stop_at_main) const {
    using namespace nest::mc::memory::util;

    auto i = 0;
    while (file_exists("backtrace_" + std::to_string(i))) {
        ++i;
    }
    auto fname = "backtrace_" + std::to_string(i);
    auto fid = std::ofstream(fname);
    for (auto& f: frames_) {
        char loc_str[64];
        snprintf(loc_str, sizeof(loc_str),"0x%lx", f.position);
        fid << loc_str << " " << f.name << "\n";
        if (stop_at_main && f.name=="main") {
            break;
        }
    }
    std::cerr << "BACKTRACE: A backtrace was generated and stored in the file " << fname << ".\n";
    std::cerr << "           View a brief summary of the backtrace by running \"scripts/print_backtrace " << fname << " -b\".\n";
    std::cerr << "           Run \"scripts/print_backtrace -h\" for more options.\n";
}

} // namespace util
} // namespace mc
} // namespace nest

#endif
