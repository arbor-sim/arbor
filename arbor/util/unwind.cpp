#include <util/unwind.hpp>

#ifdef WITH_UNWIND

#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <cxxabi.h>

#include <memory/util.hpp>
#include <util/file.hpp>

#include <cxxabi.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>

namespace arb {
namespace util {

static_assert(sizeof(std::uintptr_t)>=sizeof(unw_word_t),
        "assumption that libunwind unw_word_t can be stored in std::uintptr_t is not valid");

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

std::ostream& operator<<(std::ostream& out, const backtrace& trace) {
    for (auto& f: trace.frames_) {
        char loc_str[64];
        snprintf(loc_str, sizeof(loc_str), "0x%lx", f.position);
        out << loc_str << " " << f.name << "\n";
        if (f.name=="main") {
            break;
        }
    }
    return out;
}

#if 0
// Temporarily deprecated: automatic writing to disk of strack traces
// needs to be run-time configurable.

void backtrace::print(bool stop_at_main) const {
    using namespace arb::memory::util;

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
#endif

} // namespace util
} // namespace arb

#else

namespace arb {
namespace util {

backtrace::backtrace() {}

std::ostream& operator<<(std::ostream& out, const backtrace& trace) {
    return out;
}

//void arb::util::backtrace::print(bool) const {}

} // namespace util
} // namespace arb

#endif

