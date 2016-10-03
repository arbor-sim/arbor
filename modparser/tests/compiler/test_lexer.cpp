#include <cmath>

#include "test.hpp"
#include "../src/lexer.hpp"

//#define PRINT_LEX_STRING std::cout << "________________\n" << string << "\n________________\n";
#define PRINT_LEX_STRING

/**************************************************************
 * lexer tests
 **************************************************************/
// test identifiers
TEST(Lexer, identifiers) {
    char string[] = "_foo:\nbar, buzz f_zz";
    PRINT_LEX_STRING
    Lexer lexer(string, string+sizeof(string));

    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::identifier);
    EXPECT_EQ(t1.spelling, "_foo");
    // odds are _foo will never be a keyword
    EXPECT_EQ(is_keyword(t1), false);

    auto t2 = lexer.parse();
    EXPECT_EQ(t2.type, tok::identifier);
    EXPECT_EQ(t2.spelling, "bar");

    auto t3 = lexer.parse();
    EXPECT_EQ(t3.type, tok::comma);

    auto t4 = lexer.parse();
    EXPECT_EQ(t4.type, tok::identifier);
    EXPECT_EQ(t4.spelling, "buzz");

    auto t5 = lexer.parse();
    EXPECT_EQ(t5.type, tok::identifier);
    EXPECT_EQ(t5.spelling, "f_zz");

    auto t6 = lexer.parse();
    EXPECT_EQ(t6.type, tok::eof);
}

// test keywords
TEST(Lexer, keywords) {
    char string[] = "NEURON UNITS SOLVE else TITLE CONDUCTANCE";
    PRINT_LEX_STRING
    Lexer lexer(string, string+sizeof(string));

    // should skip all white space and go straight to eof
    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::neuron);
    EXPECT_EQ(is_keyword(t1), true);
    EXPECT_EQ(t1.spelling, "NEURON");

    auto t2 = lexer.parse();
    EXPECT_EQ(t2.type, tok::units);
    EXPECT_EQ(t2.spelling, "UNITS");

    auto t3 = lexer.parse();
    EXPECT_EQ(t3.type, tok::solve);
    EXPECT_EQ(t3.spelling, "SOLVE");

    auto t4 = lexer.parse();
    EXPECT_EQ(t4.type, tok::else_stmt);
    EXPECT_EQ(t4.spelling, "else");

    auto t5 = lexer.parse();
    EXPECT_NE(t5.type, tok::identifier);
    EXPECT_EQ(t5.spelling, "TITLE");

    auto t7 = lexer.parse();
    EXPECT_EQ(t7.type, tok::conductance);
    EXPECT_EQ(t7.spelling, "CONDUCTANCE");

    auto t6 = lexer.parse();
    EXPECT_EQ(t6.type, tok::eof);
}

// test white space
TEST(Lexer, whitespace) {
    char string[] = " \t\v\f";
    PRINT_LEX_STRING
    Lexer lexer(string, string+sizeof(string));

    // should skip all white space and go straight to eof
    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::eof);
}

// test new line
TEST(Lexer, newline) {
    char string[] = "foo \n    bar \n +\r\n-";
    PRINT_LEX_STRING
    Lexer lexer(string, string+sizeof(string));

    // get foo
    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::identifier);
    EXPECT_EQ(t1.spelling, "foo");
    EXPECT_EQ(t1.location.line, 1);
    EXPECT_EQ(t1.location.column, 1);

    auto t2 = lexer.parse();
    EXPECT_EQ(t2.type, tok::identifier);
    EXPECT_EQ(t2.spelling, "bar");
    EXPECT_EQ(t2.location.line, 2);
    EXPECT_EQ(t2.location.column, 5);

    auto t3 = lexer.parse();
    EXPECT_EQ(t3.type, tok::plus);
    EXPECT_EQ(t3.spelling, "+");
    EXPECT_EQ(t3.location.line, 3);
    EXPECT_EQ(t3.location.column, 2);

    // test for carriage return + newline, i.e. \r\n
    auto t4 = lexer.parse();
    EXPECT_EQ(t4.type, tok::minus);
    EXPECT_EQ(t4.spelling, "-");
    EXPECT_EQ(t4.location.line, 4);
    EXPECT_EQ(t4.location.column, 1);
}

// test operators
TEST(Lexer, symbols) {
    char string[] = "+-/*, t= ^ h'";
    PRINT_LEX_STRING
    Lexer lexer(string, string+sizeof(string));

    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::plus);

    auto t2 = lexer.parse();
    EXPECT_EQ(t2.type, tok::minus);

    auto t3 = lexer.parse();
    EXPECT_EQ(t3.type, tok::divide);

    auto t4 = lexer.parse();
    EXPECT_EQ(t4.type, tok::times);

    auto t5 = lexer.parse();
    EXPECT_EQ(t5.type, tok::comma);

    // test that identifier followed by = is parsed correctly
    auto t6 = lexer.parse();
    EXPECT_EQ(t6.type, tok::identifier);

    auto t7 = lexer.parse();
    EXPECT_EQ(t7.type, tok::eq);

    auto t8 = lexer.parse();
    EXPECT_EQ(t8.type, tok::pow);

    auto t9 = lexer.parse();
    EXPECT_EQ(t9.type, tok::identifier);

    // check that prime' is parsed properly after symbol
    // as this is how it is used to indicate a derivative
    auto t10 = lexer.parse();
    EXPECT_EQ(t10.type, tok::prime);

    auto t11 = lexer.parse();
    EXPECT_EQ(t11.type, tok::eof);
}

TEST(Lexer, comparison_operators) {
    char string[] = "< <= > >= == != !";
    PRINT_LEX_STRING
    Lexer lexer(string, string+sizeof(string));

    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::lt);
    auto t2 = lexer.parse();
    EXPECT_EQ(t2.type, tok::lte);
    auto t3 = lexer.parse();
    EXPECT_EQ(t3.type, tok::gt);
    auto t4 = lexer.parse();
    EXPECT_EQ(t4.type, tok::gte);
    auto t5 = lexer.parse();
    EXPECT_EQ(t5.type, tok::equality);
    auto t6 = lexer.parse();
    EXPECT_EQ(t6.type, tok::ne);
    auto t7 = lexer.parse();
    EXPECT_EQ(t7.type, tok::lnot);

    auto t8 = lexer.parse();
    EXPECT_EQ(t8.type, tok::eof);
}

// test braces
TEST(Lexer, braces) {
    char string[] = "foo}";
    PRINT_LEX_STRING
    Lexer lexer(string, string+sizeof(string));

    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::identifier);

    auto t2 = lexer.parse();
    EXPECT_EQ(t2.type, tok::rbrace);

    auto t3 = lexer.parse();
    EXPECT_EQ(t3.type, tok::eof);
}

// test comments
TEST(Lexer, comments) {
    char string[] = "foo:this is one line\nbar : another comment\n";
    PRINT_LEX_STRING
    Lexer lexer(string, string+sizeof(string));

    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::identifier);

    auto t2 = lexer.parse();
    EXPECT_EQ(t2.type, tok::identifier);
    EXPECT_EQ(t2.spelling, "bar");
    EXPECT_EQ(t2.location.line, 2);

    auto t3 = lexer.parse();
    EXPECT_EQ(t3.type, tok::eof);
}

// test numbers
TEST(Lexer, numbers) {
    char string[] = "1 .3 23 87.99 12. -3";
    PRINT_LEX_STRING
    Lexer lexer(string, string+sizeof(string));

    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::number);
    EXPECT_EQ(std::stod(t1.spelling), 1.0);

    auto t2 = lexer.parse();
    EXPECT_EQ(t2.type, tok::number);
    EXPECT_EQ(std::stod(t2.spelling), 0.3);

    auto t3 = lexer.parse();
    EXPECT_EQ(t3.type, tok::number);
    EXPECT_EQ(std::stod(t3.spelling), 23.0);

    auto t4 = lexer.parse();
    EXPECT_EQ(t4.type, tok::number);
    EXPECT_EQ(std::stod(t4.spelling), 87.99);

    auto t5 = lexer.parse();
    EXPECT_EQ(t5.type, tok::number);
    EXPECT_EQ(std::stod(t5.spelling), 12.0);

    // the lexer does not decide where the - sign goes
    // the parser uses additional contextual information to
    // decide if the minus is a binary or unary expression
    auto t6 = lexer.parse();
    EXPECT_EQ(t6.type, tok::minus);

    auto t7 = lexer.parse();
    EXPECT_EQ(t7.type, tok::number);
    EXPECT_EQ(std::stod(t7.spelling), 3.0);

    auto t8 = lexer.parse();
    EXPECT_EQ(t8.type, tok::eof);
}


