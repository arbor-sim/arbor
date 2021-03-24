#pragma once

#include <memory>
#include <ostream>
#include <string>

namespace arborio {

namespace asc {

struct src_location {
    unsigned line = 0;
    unsigned column = 0;

    src_location() = default;

    src_location(unsigned l, unsigned c):
        line(l), column(c)
    {}
};

std::ostream& operator<<(std::ostream& o, const src_location& l);

enum class tok {
    lparen,     // left parenthesis '('
    rparen,     // right parenthesis ')'
    lt,         // less than '<'
    gt,         // less than '>'
    comma,      // comma ','
    real,       // real number
    integer,    // integer
    symbol,     // symbol
    string,     // string, written as "spelling"
    pipe,       // pipe '|'
    eof,        // end of file/input
    error       // special error state marker
};

std::ostream& operator<<(std::ostream&, const tok&);

struct token {
    src_location loc;
    tok kind;
    std::string spelling;
};

std::ostream& operator<<(std::ostream&, const token&);

// Forward declare pimpled implementation.
class lexer_impl;

class lexer {
public:
    lexer(const char* begin);

    const token& current();
    const token& next(unsigned n=1);
    token peek(unsigned n=1);

    ~lexer();

private:
    std::unique_ptr<lexer_impl> impl_;
};

} // namespace asc

} // namespace arborio
