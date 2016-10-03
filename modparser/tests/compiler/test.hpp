#pragma once

#include <gtest.h>

#include "../src/parser.hpp"
#include "../src/util.hpp"

//#define VERBOSE_TEST
#ifdef VERBOSE_TEST
#define VERBOSE_PRINT(x) std::cout << (x) << std::endl;
#else
#define VERBOSE_PRINT(x)
#endif

static expression_ptr parse_line_expression(std::string const& s) {
    return Parser(s).parse_line_expression();
}

static expression_ptr parse_expression(std::string const& s) {
    return Parser(s).parse_expression();
}

static expression_ptr parse_function(std::string const& s) {
    return Parser(s).parse_function();
}

static expression_ptr parse_procedure(std::string const& s) {
    return Parser(s).parse_procedure();
}

