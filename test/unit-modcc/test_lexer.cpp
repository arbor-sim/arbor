#include <cctype>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <utility>

#include "common.hpp"
#include "lexer.hpp"

class VerboseLexer: public Lexer {
public:
    template <typename... Args>
    VerboseLexer(Args&&... args): Lexer(std::forward<Args>(args)...) {
        verbose_print("________________");
        verbose_print(std::string(begin_, end_));
        verbose_print("________________");
    }

    Token parse() {
        auto tok = Lexer::parse();
        verbose_print("token: ",tok);
        return tok;
    }

    char character() {
        char c = Lexer::character();
        verbose_print("character: ", pretty(c));
        return c;
    }

    static const char* pretty(char c) {
        static char buf[5] = "XXXX";
        if (!std::isprint(c)) {
            snprintf(buf, sizeof buf, "0x%02x", (unsigned)c);
        }
        else {
            buf[0] = c;
            buf[1] = 0;
        }
        return buf;
    }
};

/**************************************************************
 * lexer tests
 **************************************************************/
// test identifiers
TEST(Lexer, identifiers) {
    char string[] = "_foo:\nbar, buzz f_zz";
    VerboseLexer lexer(string, string+sizeof(string));

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
    char string[] = "NEURON UNITS SOLVE else TITLE CONDUCTANCE KINETIC CONSERVE LOCAL";
    VerboseLexer lexer(string, string+sizeof(string));

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

    auto t6 = lexer.parse();
    EXPECT_EQ(t6.type, tok::conductance);
    EXPECT_EQ(t6.spelling, "CONDUCTANCE");

    auto t7 = lexer.parse();
    EXPECT_EQ(t7.type, tok::kinetic);
    EXPECT_EQ(t7.spelling, "KINETIC");

    auto t8 = lexer.parse();
    EXPECT_EQ(t8.type, tok::conserve);
    EXPECT_EQ(t8.spelling, "CONSERVE");

    auto t9 = lexer.parse();
    EXPECT_EQ(t9.type, tok::local);
    EXPECT_EQ(t9.spelling, "LOCAL");

    auto tlast = lexer.parse();
    EXPECT_EQ(tlast.type, tok::eof);
}

// test white space
TEST(Lexer, whitespace) {
    char string[] = " \t\v\f";
    VerboseLexer lexer(string, string+sizeof(string));

    // should skip all white space and go straight to eof
    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::eof);
}

// test new line
TEST(Lexer, newline) {
    char string[] = "foo \n    bar \n +\r\n-";
    VerboseLexer lexer(string, string+sizeof(string));

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
    char string[] = "+-/*, t= ^ h'<->~";
    VerboseLexer lexer(string, string+sizeof(string));

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
    EXPECT_EQ(t11.type, tok::arrow);

    auto t12 = lexer.parse();
    EXPECT_EQ(t12.type, tok::tilde);

    auto tlast = lexer.parse();
    EXPECT_EQ(tlast.type, tok::eof);
}

TEST(Lexer, comparison_operators) {
    char string[] = "< <= > >= == != ! && ||";
    VerboseLexer lexer(string, string+sizeof(string));

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
    EXPECT_EQ(t8.type, tok::land);
    auto t9 = lexer.parse();
    EXPECT_EQ(t9.type, tok::lor);
    auto t10 = lexer.parse();
    EXPECT_EQ(t10.type, tok::eof);
}

// test braces
TEST(Lexer, braces) {
    char string[] = "foo}";
    VerboseLexer lexer(string, string+sizeof(string));

    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::identifier);

    auto t2 = lexer.parse();
    EXPECT_EQ(t2.type, tok::rbrace);

    auto t3 = lexer.parse();
    EXPECT_EQ(t3.type, tok::eof);
}

// test comments
TEST(Lexer, comments) {
    char string[] = "foo:this is one line\n"
                    "bar : another comment\n"
                    "foobar ? another comment\n";
    VerboseLexer lexer(string, string+sizeof(string));

    auto t1 = lexer.parse();
    EXPECT_EQ(t1.type, tok::identifier);

    auto t2 = lexer.parse();
    EXPECT_EQ(t2.type, tok::identifier);
    EXPECT_EQ(t2.spelling, "bar");
    EXPECT_EQ(t2.location.line, 2);

    auto t3 = lexer.parse();
    EXPECT_EQ(t3.type, tok::identifier);
    EXPECT_EQ(t3.spelling, "foobar");
    EXPECT_EQ(t3.location.line, 3);

    auto t4 = lexer.parse();
    EXPECT_EQ(t4.type, tok::eof);
}

// test numbers
TEST(Lexer, numbers) {
    auto numeric = [](tok t) { return t==tok::real || t==tok::integer; };
    std::istringstream floats_stream("1 23 .3 87.99 12. 1.e3 1.2e+2 23e-3 -3");

    std::vector<double> floats;
    std::copy(std::istream_iterator<double>(floats_stream),
              std::istream_iterator<double>(),
              std::back_inserter(floats));

    // hand-parse these ...
    std::vector<long long> check_ints = {1, 23, 3};
    std::vector<long long> ints;

    VerboseLexer lexer(floats_stream.str());
    auto t = lexer.parse();
    auto iter = floats.cbegin();
    while (t.type != tok::eof && iter != floats.cend()) {
        EXPECT_EQ(lexerStatus::happy, lexer.status());
        if (*iter < 0) {
            // the lexer does not decide where the - sign goes
            // the parser uses additional contextual information to
            // decide if the minus is a binary or unary expression
            EXPECT_EQ(tok::minus, t.type);
            t = lexer.parse();
            EXPECT_TRUE(numeric(t.type));
            if (t.type==tok::integer) ints.push_back(std::stoll(t.spelling));
            EXPECT_EQ(-(*iter), std::stod(t.spelling));
        }
        else {
            EXPECT_TRUE(numeric(t.type));
            if (t.type==tok::integer) ints.push_back(std::stoll(t.spelling));
            EXPECT_EQ(*iter, std::stod(t.spelling));
        }

        ++iter;
        t = lexer.parse();
    }

    EXPECT_EQ(floats.cend(), iter);
    EXPECT_EQ(tok::eof, t.type);
    EXPECT_EQ(check_ints, ints);

    // check case where 'E' is not followed by +, -, or a digit explicitly
    lexer = VerboseLexer("7.2E");
    t = lexer.parse();
    EXPECT_EQ(lexerStatus::happy, lexer.status());
    EXPECT_EQ(tok::real, t.type);
    EXPECT_EQ(t.spelling, "7.2");
    EXPECT_EQ(lexer.character(), 'E');

    lexer = VerboseLexer("3E+E2");
    t = lexer.parse();
    EXPECT_EQ(lexerStatus::happy, lexer.status());
    EXPECT_EQ(tok::integer, t.type);
    EXPECT_EQ(t.spelling, "3");
    EXPECT_EQ(lexer.character(), 'E');
    EXPECT_EQ(lexer.character(), '+');

    // 'bad' numbers should give errors
    lexer = VerboseLexer("1.2.3");
    lexer.parse();
    EXPECT_EQ(lexerStatus::error, lexer.status());

    lexer = VerboseLexer("1.2E4.3");
    lexer.parse();
    EXPECT_EQ(lexerStatus::error, lexer.status());

    // single or triple & or | should give errors
    lexer = VerboseLexer("&");
    lexer.parse();
    EXPECT_EQ(lexerStatus::error, lexer.status());

    lexer = VerboseLexer("&&&");
    lexer.parse();
    EXPECT_EQ(lexerStatus::error, lexer.status());

    lexer = VerboseLexer("|");
    lexer.parse();
    EXPECT_EQ(lexerStatus::error, lexer.status());

    lexer = VerboseLexer("|||");
    lexer.parse();
    EXPECT_EQ(lexerStatus::error, lexer.status());
}
