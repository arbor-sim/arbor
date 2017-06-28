//
// AVX2 backend
//

#pragma once

#include "backends/base.hpp"


namespace nest {
namespace mc {
namespace modcc {

// Specialize for the different architectures
template<>
struct simd_intrinsics<targetKind::avx2> {
    static bool has_scatter() {
        return false;
    }

    static bool has_gather() {
        return true;
    }

    static std::string emit_headers() {
        return "#include <immintrin.h>";
    };

    static std::string emit_simd_width() {
        return "256";
    }

    static std::string emit_value_type() {
        return "__m256d";
    }

    static std::string emit_index_type() {
        return "__m128i";
    }

    template<typename T1, typename T2>
    static void emit_binary_op(TextBuffer& tb, tok op,
                               const T1& arg1, const T2& arg2) {
        switch (op) {
        case tok::plus:
            tb << "_mm256_add_pd(";
            break;
        case tok::minus:
            tb << "_mm256_sub_pd(";
            break;
        case tok::times:
            tb << "_mm256_mul_pd(";
            break;
        case tok::divide:
            tb << "_mm256_div_pd(";
            break;
        default:
            throw std::invalid_argument("Unknown binary operator");
        }

        emit_operands(tb, arg_emitter(arg1), arg_emitter(arg2));
        tb << ")";
    }

    template<typename T>
    static void emit_unary_op(TextBuffer& tb, tok op, const T& arg) {
        switch (op) {
        case tok::minus:
            tb << "_mm256_sub_pd(_mm256_set1_pd(0), ";
            break;
        case tok::exp:
            tb << "_mm256_exp_pd(";
            break;
        case tok::log:
            tb << "_mm256_log_pd(";
            break;
        default:
            throw std::invalid_argument("Unknown unary operator");
        }

        emit_operands(tb, arg_emitter(arg));
        tb << ")";
    }

    template<typename B, typename E>
    static void emit_pow(TextBuffer& tb, const B& base, const E& exp) {
        tb << "_mm256_pow_pd(";
        emit_operands(tb, arg_emitter(base), arg_emitter(exp));
        tb << ")";
    }

    template<typename A, typename V>
    static void emit_store_unaligned(TextBuffer& tb, const A& addr,
                                     const V& value) {
        tb << "_mm256_storeu_pd(";
        emit_operands(tb, arg_emitter(addr), arg_emitter(value));
        tb << ")";
    }

    template<typename A>
    static void emit_load_unaligned(TextBuffer& tb, const A& addr) {
        tb << "_mm256_loadu_pd(";
        emit_operands(tb, arg_emitter(addr));
        tb << ")";
    }

    template<typename A>
    static void emit_load_index(TextBuffer& tb, const A& addr) {
        tb << "_mm_lddqu_si128(";
        emit_operands(tb, arg_emitter(addr));
        tb << ")";
    }

    template<typename A, typename I, typename V, typename S>
    static void emit_scatter(TextBuffer& tb, const A& addr,
                             const I& index, const V& value, const S& scale) {
        // no support of scatter in AVX2, so revert to simple scalar updates
        std::string scalar_index_ptr = varprefix + std::to_string(varcnt++);
        std::string scalar_value_ptr = varprefix + std::to_string(varcnt++);

        tb.end_line("{");
        tb.increase_indentation();

        // FIXME: should probably read "index_type*"
        tb.add_gutter();
        tb << "int* " << scalar_index_ptr
           << " = (int*) &" << index;
        tb.end_line(";");

        tb.add_gutter();
        tb << "value_type* " << scalar_value_ptr
           << " = (value_type*) &" << value;
        tb.end_line(";");

        tb.add_line("for (int k_ = 0; k_ < simd_width; ++k_) {");
        tb.increase_indentation();
        tb.add_gutter();
        tb << addr << "[" << scalar_index_ptr << "[k_]] = "
           << scalar_value_ptr << "[k_]";
        tb.end_line(";");

        tb.decrease_indentation();
        tb.add_line("}");

        tb.decrease_indentation();
        tb.add_gutter();
        tb << "}";
    }

    template<typename A, typename I, typename S>
    static void emit_gather(TextBuffer& tb, const A& addr,
                            const I& index, const S& scale) {
        tb << "_mm256_i32gather_pd(";
        emit_operands(tb, arg_emitter(addr), arg_emitter(index),
                      arg_emitter(scale));
        tb << ")";
    }

    // In avx2 require 4-wide gather of i32 indices.
    template<typename A, typename I, typename S>
    static void emit_gather_index(TextBuffer& tb, const A& addr,
                                  const I& index, const S& scale) {
        tb << "_mm_i32gather_epi32(";
        emit_operands(tb, arg_emitter(addr), arg_emitter(index),
                      arg_emitter(scale));
        tb << ")";
    }

    template<typename T>
    static void emit_set_value(TextBuffer& tb, const T& arg) {
        tb << "_mm256_set1_pd(";
        emit_operands(tb, arg_emitter(arg));
        tb << ")";
    }

private:
    static int varcnt;
    const static std::string varprefix;
};

int simd_intrinsics<targetKind::avx2>::varcnt = 0;
const std::string simd_intrinsics<targetKind::avx2>::varprefix = "_r";

}}} // closing namespaces
