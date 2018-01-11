//
// AVX512 backend
//

#pragma once

#include "backends/base.hpp"

namespace modcc {

// Specialize for the different architectures
template<>
struct simd_intrinsics<simdKind::avx512> {

    static bool has_scatter() {
        return true;
    }

    static bool has_gather() {
        return true;
    }

    static std::string emit_headers() {
        return
            "#include <immintrin.h>\n"
            "#include <backends/multicore/intrin.hpp>";
    };

    static std::string emit_simd_width() {
        return "512";
    }

    static std::string emit_value_type() {
        return "vecd_avx512";
    }

    static std::string emit_index_type() {
        return "__m256i";
    }

    template<typename T1, typename T2>
    static void emit_binary_op(TextBuffer& tb, tok op,
                               const T1& arg1, const T2& arg2) {
        switch (op) {
        case tok::plus:
            tb << "add(";
            break;
        case tok::minus:
            tb << "sub(";
            break;
        case tok::times:
            tb << "mul(";
            break;
        case tok::divide:
            tb << "div(";
            break;
        case tok::max:
            tb << "max(";
            break;
        case tok::min:
            tb << "min(";
            break;
        default:
            throw std::invalid_argument("Unable to generate avx512 for binary operator " + token_map[op]);
        }

        emit_operands(tb, arg_emitter(arg1), arg_emitter(arg2));
        tb << ")";
    }

    template<typename T>
    static void emit_unary_op(TextBuffer& tb, tok op, const T& arg) {
        switch (op) {
        case tok::minus:
            tb << "sub(set(0), ";
            break;
        case tok::exp:
            tb << "_mm512_exp_pd(";
            break;
        case tok::log:
            tb << "_mm512_log_pd(";
            break;
        case tok::abs:
            tb << "abs(";
            break;
        case tok::exprelr:
            tb << "exprelr(";
            break;
        default:
            throw std::invalid_argument("Unable to generate avx512 for unary operator " + token_map[op]);
        }

        emit_operands(tb, arg_emitter(arg));
        tb << ")";
    }

    template<typename B, typename E>
    static void emit_pow(TextBuffer& tb, const B& base, const E& exp) {
        tb << "_mm512_pow_pd(";
        emit_operands(tb, arg_emitter(base), arg_emitter(exp));
        tb << ")";
    }

    template<typename A, typename V>
    static void emit_store_unaligned(TextBuffer& tb, const A& addr,
                                     const V& value) {
        tb << "_mm512_storeu_pd(";
        emit_operands(tb, arg_emitter(addr), arg_emitter(value));
        tb << ")";
    }

    template<typename A>
    static void emit_load_unaligned(TextBuffer& tb, const A& addr) {
        tb << "_mm512_loadu_pd(";
        emit_operands(tb, arg_emitter(addr));
        tb << ")";
    }

    template<typename A>
    static void emit_load_index(TextBuffer& tb, const A& addr) {
        tb << "_mm256_lddqu_si256(";
        emit_operands(tb, arg_emitter(addr));
        tb << ")";
    }

    template<typename A, typename I, typename V, typename S>
    static void emit_scatter(TextBuffer& tb, const A& addr,
                             const I& index, const V& value, const S& scale) {
        tb << "_mm512_i32scatter_pd(";
        emit_operands(tb, arg_emitter(addr), arg_emitter(index),
                      arg_emitter(value), arg_emitter(scale));
        tb << ")";
    }

    template<typename A, typename I, typename S>
    static void emit_gather(TextBuffer& tb, const A& addr,
                            const I& index, const S& scale) {
        tb << "_mm512_i32gather_pd(";
        emit_operands(tb, arg_emitter(index), arg_emitter(addr),
                      arg_emitter(scale));
        tb << ")";
    }

    // In avx512 require 8-wide gather of i32 indices.
    template<typename A, typename I, typename S>
    static void emit_gather_index(TextBuffer& tb, const A& addr,
                                  const I& index, const S& scale) {
        tb << "_mm256_i32gather_epi32(";
        emit_operands(tb, arg_emitter(addr), arg_emitter(index),
                      arg_emitter(scale));
        tb << ")";
    }

    template<typename T>
    static void emit_set_value(TextBuffer& tb, const T& arg) {
        tb << "set(";
        emit_operands(tb, arg_emitter(arg));
        tb << ")";
    }
};

} // namespace modcc;
