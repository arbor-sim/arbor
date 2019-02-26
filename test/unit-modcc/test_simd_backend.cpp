#if 0
// Disabled pending new SIMD printer code.

#include "backends/simd.hpp"
#include "textbuffer.hpp"
#include "token.hpp"
#include "common.hpp"


TEST(avx512, emit_binary_op) {
    TextBuffer tb;

    using simd_backend = modcc::simd_intrinsics<simdKind::avx512>;

    simd_backend::emit_binary_op(tb, tok::plus, "a", "b");
    EXPECT_EQ("_mm512_add_pd(a, b)", tb.str());

    // Test also lambdas
    std::string lhs = "a";
    std::string rhs = "b";
    tb.clear();
    simd_backend::emit_binary_op(tb, tok::minus,
                                 [lhs](TextBuffer& tb) { tb << lhs; },
                                 [rhs](TextBuffer& tb) { tb << rhs; });
    EXPECT_EQ("_mm512_sub_pd(a, b)", tb.str());


    // Test mixed: lambdas + strings
    tb.clear();
    simd_backend::emit_binary_op(tb, tok::times,
                                 [lhs](TextBuffer& tb) { tb << lhs; }, "b");
    EXPECT_EQ("_mm512_mul_pd(a, b)", tb.str());

    tb.clear();
    simd_backend::emit_binary_op(tb, tok::divide, "a", "b");
    EXPECT_EQ("_mm512_div_pd(a, b)", tb.str());


    tb.clear();
    simd_backend::emit_pow(tb, "a", "b");
    EXPECT_EQ("_mm512_pow_pd(a, b)", tb.str());
}

TEST(avx512, emit_unary_op) {
    TextBuffer tb;

    using simd_backend = modcc::simd_intrinsics<simdKind::avx512>;

    // Test lambdas for generating the argument
    std::string arg = "a";
    simd_backend::emit_unary_op(tb, tok::minus,
                                [arg](TextBuffer& tb) { tb << arg; });
    EXPECT_EQ("_mm512_sub_pd(_mm512_set1_pd(0), a)", tb.str());

    tb.clear();
    simd_backend::emit_unary_op(tb, tok::exp, "a");
    EXPECT_EQ("_mm512_exp_pd(a)", tb.str());

    tb.clear();
    simd_backend::emit_unary_op(tb, tok::log, "a");
    EXPECT_EQ("_mm512_log_pd(a)", tb.str());

    tb.clear();
    simd_backend::emit_load_index(tb, "&a");
    EXPECT_EQ("_mm256_lddqu_si256(&a)", tb.str());
}
#endif
