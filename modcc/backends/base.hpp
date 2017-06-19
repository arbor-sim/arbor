//
// Base SIMD backend functionality
//

#pragma once

#include <functional>
#include <stdexcept>
#include <string>

#include "options.hpp"
#include "token.hpp"
#include "textbuffer.hpp"
#include "util/meta.hpp"

namespace nest {
namespace mc {
namespace modcc {

using nest::mc::util::enable_if_t;
using operand_fn_t = std::function<void(TextBuffer&)>;

static void emit_operands(TextBuffer& tb, operand_fn_t emitter) {
    emitter(tb);
}

template<typename ... Args>
static void emit_operands(TextBuffer& tb, operand_fn_t emitter, Args ... args) {
    emitter(tb);
    tb << ", ";
    emit_operands(tb, args ...);
}

template<typename T>
static enable_if_t<!std::is_convertible<T, operand_fn_t>::value, operand_fn_t>
arg_emitter(const T& arg) {
    return [arg](TextBuffer& tb) { tb << arg; };
}

static operand_fn_t arg_emitter(const operand_fn_t& arg) {
    return arg;
}


template<targetKind Arch>
struct simd_intrinsics {
    static std::string emit_headers();
    static std::string emit_simd_width();
    static std::string emit_simd_value_type();
    static std::string emit_simd_index_type();

    template<typename T1, typename T2>
    static void emit_binary_op(TextBuffer& tb, tok op,
                               const T1& arg1, const T2& arg2);

    template<typename T>
    static void emit_unary_op(TextBuffer& tb, tok op, const T& arg);

    template<typename B, typename E>
    static void emit_pow(TextBuffer& tb, const B& base, const E& exp);

    template<typename A, typename V>
    static void emit_store_unaligned(TextBuffer& tb, const A& addr, const V& value);

    template<typename A>
    static void emit_load_unaligned(TextBuffer& tb, const A& addr);

    template<typename A>
    static void emit_load_index(TextBuffer& tb, const A& addr);

    template<typename A, typename I, typename V, typename S>
    static void emit_scatter(TextBuffer& tb, const A& addr,
                             const I& index, const V& value, const S& scale);

    template<typename A, typename I, typename S>
    static void emit_gather(TextBuffer& tb, const A& addr,
                            const I& index, const S& scale);

    // int32 value version of `emit_gather` to look up cell indices
    template<typename A, typename I, typename S>
    static void emit_gather_index(TextBuffer& tb, const A& addr,
                                  const I& index, const S& scale);

    template<typename T>
    static void emit_set_value(TextBuffer& tb, const T& arg);

    static bool has_gather();
    static bool has_scatter();
};

}}} // closing namespaces
