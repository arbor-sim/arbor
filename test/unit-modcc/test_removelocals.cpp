#include <iostream>
#include <map>
#include <regex>
#include <string>

#include "expression.hpp"
#include "solvers.hpp"
#include "parser.hpp"
#include "scope.hpp"

#include "common.hpp"

symbol_ptr make_global(std::string name) {
    return make_symbol<VariableExpression>(Location(), std::move(name));
}

using symbol_map = Scope<Symbol>::symbol_map;

Symbol* add_procedure(symbol_map& symbols, const char* src) {
    auto proc = Parser(src).parse_procedure();
    std::string name = proc->is_procedure()->name();

    Symbol* weak = (symbols[name] = std::move(proc)).get();
    weak->semantic(symbols);
    return weak;
}

Symbol* add_global(symbol_map& symbols, const std::string& name) {
    auto& var = (symbols[name] = make_symbol<VariableExpression>(Location(), name));
    return var.get();
}

TEST(remove_unused_locals, simple) {
    const char* before_src =
    "PROCEDURE before {  \n"
    "    LOCAL a      \n"
    "    LOCAL b      \n"
    "    a = 3        \n"
    "    g = a        \n"
    "    b = 4        \n"
    "}                \n";

    const char* expected_src =
    "PROCEDURE expected {  \n"
    "    LOCAL a      \n"
    "    a = 3        \n"
    "    g = a        \n"
    "}                \n";

    symbol_map symbols;
    add_global(symbols, "g");

    auto before = add_procedure(symbols, before_src);
    auto before_body = before->is_procedure()->body();

    auto expected = add_procedure(symbols, expected_src);
    auto expected_body = expected->is_procedure()->body();

    auto after = remove_unused_locals(before_body);

    verbose_print("before: ", before_body);
    verbose_print("after: ", after);
    verbose_print("expected: ", expected_body);

    EXPECT_EXPR_EQ(expected_body, after.get());
}

TEST(remove_unused_locals, compound) {
    const char* before_src =
    "PROCEDURE before {   \n"
    "    LOCAL a, b, c, d \n"
    "    g1 = a           \n"
    "    g2 = c           \n"
    "}                    \n";

    const char* expected_src =
    "PROCEDURE expected { \n"
    "    LOCAL a, c       \n"
    "    g1 = a           \n"
    "    g2 = c           \n"
    "}                    \n";

    symbol_map symbols;
    add_global(symbols, "g1");
    add_global(symbols, "g2");

    auto before = add_procedure(symbols, before_src);
    auto before_body = before->is_procedure()->body();

    auto expected = add_procedure(symbols, expected_src);
    auto expected_body = expected->is_procedure()->body();

    auto after = remove_unused_locals(before_body);

    verbose_print("before: ", before_body);
    verbose_print("after: ", after);
    verbose_print("expected: ", expected_body);

    EXPECT_EXPR_EQ(expected_body, after.get());
}

TEST(remove_unused_locals, with_dependencies) {
    const char* before_src =
    "PROCEDURE before {  \n"
    "    LOCAL a      \n"
    "    LOCAL b      \n"
    "    LOCAL c      \n"
    "    LOCAL d      \n"
    "    LOCAL e      \n"
    "    LOCAL f      \n"
    "    g1 = f       \n"
    "    b = 3        \n"
    "    a = b-e      \n"
    "    c = log(a)+2 \n"
    "    d = 4        \n"
    "    g2 = c       \n"
    "}                \n";

    const char* expected_src =
    "PROCEDURE expected {  \n"
    "    LOCAL a      \n"
    "    LOCAL b      \n"
    "    LOCAL c      \n"
    "    LOCAL e      \n"
    "    LOCAL f      \n"
    "    g1 = f       \n"
    "    b = 3        \n"
    "    a = b-e      \n"
    "    c = log(a)+2 \n"
    "    g2 = c       \n"
    "}                \n";

    symbol_map symbols;
    add_global(symbols, "g1");
    add_global(symbols, "g2");

    auto before = add_procedure(symbols, before_src);
    auto before_body = before->is_procedure()->body();

    auto expected = add_procedure(symbols, expected_src);
    auto expected_body = expected->is_procedure()->body();

    auto after = remove_unused_locals(before_body);

    verbose_print("before: ", before_body);
    verbose_print("after: ", after);
    verbose_print("expected: ", expected_body);

    EXPECT_EXPR_EQ(expected_body, after.get());
}

TEST(remove_unused_locals, inner_block) {
    const char* before_src =
    "PROCEDURE before {  \n"
    "    LOCAL a      \n"
    "    LOCAL b      \n"
    "    LOCAL c      \n"
    "    LOCAL d      \n"
    "    if (a>0) {   \n"
    "        g = b    \n"
    "    }            \n"
    "    else {       \n"
    "        d = g    \n"
    "        g = c    \n"
    "    }            \n"
    "}                \n";

    const char* expected_src =
    "PROCEDURE expected {  \n"
    "    LOCAL a      \n"
    "    LOCAL b      \n"
    "    LOCAL c      \n"
    "    if (a>0) {   \n"
    "        g = b    \n"
    "    }            \n"
    "    else {       \n"
    "        g = c    \n"
    "    }            \n"
    "}                \n";

    symbol_map symbols;
    add_global(symbols, "g");

    auto before = add_procedure(symbols, before_src);
    auto before_body = before->is_procedure()->body();

    auto expected = add_procedure(symbols, expected_src);
    auto expected_body = expected->is_procedure()->body();

    auto after = remove_unused_locals(before_body);

    verbose_print("before: ", before_body);
    verbose_print("after: ", after);
    verbose_print("expected: ", expected_body);

    EXPECT_EXPR_EQ(expected_body, after.get());
}
