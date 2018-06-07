#include <regex>
#include <string>

#include "common.hpp"

bool g_verbose_flag = false;

std::string plain_text(Expression* expr) {
    static std::regex csi_code(R"_(\x1B\[.*?[\x40-\x7E])_");

    // We need to pass a std::string as a third argument; intel compiler
    // complains otherwise
    return !expr? "null": regex_replace(
        expr->to_string(), csi_code, std::string(""));
}

::testing::AssertionResult assert_expr_eq(const char *arg1, const char *arg2, Expression* expected, Expression* value) {
    auto value_rep = plain_text(value);
    auto expected_rep = plain_text(expected);

    if ((!value && expected) || (value && !expected) || (value_rep!=expected_rep)) {
        return ::testing::AssertionFailure()
                    << "Value of: " << arg2 << "\n"
                    << "  Actual: " << value_rep << "\n"
                    << "Expected: " << arg1 << "\n"
                    << "Which is: " << expected_rep << "\n";
    }
    else {
        return ::testing::AssertionSuccess();
    }
}
