#pragma once

// Internal TRACE macros and formatters for debugging during
// development.

#include <iostream>
#include <mutex>

// Required for endianness macros:
#include <sys/types.h>

#include "io/locked_ostream.hpp"
#include "io/sepval.hpp"
#include "io/serialize_hex.hpp"


// TRACE(expr1 [, expr2 ...])
//
// Emit current source location to std::cerr, followed by the
// literal expressions expr1, ..., and then the values of those expressions.
//
// TRACE output is to std::cerr is serialized.

#define TRACE(vars...) arb::impl::debug_emit_trace(__FILE__, __LINE__, #vars, ##vars)


// DEBUG << ...;
//
// Emit arguments to std::cerr followed by a newline.
// DEBUG output to std::cerr is serialized.

#define DEBUG arb::impl::emit_nl_locked(std::cerr.rdbuf())


namespace arb {

namespace impl {
    inline void debug_emit_csv(std::ostream&) {}

    template <typename Head, typename... Tail>
    void debug_emit_csv(std::ostream& out, const Head& head, const Tail&... tail) {
        out << head;
        if (sizeof...(tail)) {
            out << ", ";
        }
        debug_emit_csv(out, tail...);
    }

    inline void debug_emit_trace_leader(std::ostream& out, const char* file, int line, const char* vars) {
        out << file << ':' << line << ": " << vars << ": ";
    }

    struct emit_nl_locked: public io::locked_ostream {
        emit_nl_locked(std::streambuf* buf):
            io::locked_ostream(buf),
            lock_(this->guard())
        {}

        ~emit_nl_locked() {
            if (rdbuf()) {
                (*this) << std::endl;
            }
        }

    private:
        std::unique_lock<std::mutex> lock_;
    };

    template <typename... Args>
    void debug_emit_trace(const char* file, int line, const char* varlist, const Args&... args) {
        impl::emit_nl_locked out(std::cerr.rdbuf());

        out.precision(17);
        impl::debug_emit_trace_leader(out, file, line, varlist);
        impl::debug_emit_csv(out, args...);
    }
} // namespace impl

} // namespace arb
