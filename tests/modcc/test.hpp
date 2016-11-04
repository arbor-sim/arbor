#pragma once

#include <gtest.h>

#include "parser.hpp"
#include "modccutil.hpp"

extern bool g_verbose_flag;

#define VERBOSE_PRINT(x) (g_verbose_flag && std::cout << (x) << "\n")

inline expression_ptr parse_line_expression(std::string const& s) {
    return Parser(s).parse_line_expression();
}

inline expression_ptr parse_expression(std::string const& s) {
    return Parser(s).parse_expression();
}

inline expression_ptr parse_function(std::string const& s) {
    return Parser(s).parse_function();
}

inline expression_ptr parse_procedure(std::string const& s) {
    return Parser(s).parse_procedure();
}
