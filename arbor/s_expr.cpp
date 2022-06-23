#include <cctype>
#include <cstring>
#include <string>
#include <memory>
#include <unordered_map>
#include <ostream>
#include <variant>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/s_expr.hpp>

#include "util/strprintf.hpp"

namespace arb {

inline bool is_alphanumeric(char c) {
    return std::isdigit(c) || std::isalpha(c);
}
inline bool is_plusminus(char c) {
    return (c=='-' || c=='+');
}
inline bool is_valid_symbol_char(char c) {
    switch (c) {
        case '+':
        case '-':
        case '*':
        case '/':
        case '@':
        case '$':
        case '%':
        case '^':
        case '&':
        case '_':
        case '=':
        case '<':
        case '>':
        case '~':
        case '.':
            return true;
        default:
            return is_alphanumeric(c);
    }
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const src_location& l) {
    return o << l.line << ":" << l.column;
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const tok& t) {
    switch (t) {
        case tok::nil:    return o << "nil";
        case tok::lparen: return o << "lparen";
        case tok::rparen: return o << "rparen";
        case tok::real:   return o << "real";
        case tok::integer:return o << "integer";
        case tok::symbol: return o << "symbol";
        case tok::string: return o << "string";
        case tok::eof:    return o << "eof";
        case tok::error:  return o << "error";
    }
    return o << "<unknown>";
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const token& t) {
    if (t.kind==tok::string) {
        return o << util::pprintf("\"{}\"", t.spelling);
    }
    return o << util::pprintf("{}", t.spelling);
}

//
// lexer
//

struct s_expr_lexer_error: public arb::arbor_internal_error {
    s_expr_lexer_error(const std::string& msg, src_location l):
        arbor_internal_error(util::pprintf("s-expression internal error at {}: {}", l, msg))
    {}
};

static std::unordered_map<tok, const char*> tok_to_keyword = {
    {tok::nil,    "nil"},
};

static std::unordered_map<std::string, tok> keyword_to_tok = {
    {"nil",    tok::nil},
};

class lexer {
    const char* line_start_;
    const char* stream_;
    unsigned line_;
    token token_;

public:

    lexer(const char* begin):
        line_start_(begin), stream_(begin), line_(0)
    {
        // Prime the first token.
        parse();
    }

    // Return the current token in the stream.
    const token& current() {
        return token_;
    }

    const token& next() {
        parse();
        return token_;
    }

private:

    src_location loc() const {
        return src_location(line_+1, stream_-line_start_+1);
    }

    bool empty() const {
        return *stream_ == '\0';
    }

    // Consume and return the next token in the stream.
    void parse() {
        using namespace std::string_literals;
#define ARB_CASE_LETTERS                                                                           \
        case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g': case 'h': case 'i':  \
        case 'j': case 'k': case 'l': case 'm': case 'n': case 'o': case 'p': case 'q': case 'r':  \
        case 's': case 't': case 'u': case 'v': case 'w': case 'x': case 'y': case 'z':            \
        case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G': case 'H': case 'I':  \
        case 'J': case 'K': case 'L': case 'M': case 'N': case 'O': case 'P': case 'Q': case 'R':  \
        case 'S': case 'T': case 'U': case 'V': case 'W': case 'X': case 'Y': case 'Z':
#define ARB_CASE_DIGITS                                                                            \
        case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8':  \
        case '9':

        while (!empty()) {
            switch (*stream_) {
                // white space
                case ' '    :
                case '\t'   :
                case '\v'   :
                case '\f'   :
                    ++stream_;
                    continue;   // skip to next character

                // new line
                case '\n'   :
                    line_++;
                    ++stream_;
                    line_start_ = stream_;
                    continue;

                // carriage return (windows new line)
                case '\r'   :
                    ++stream_;
                    if(*stream_ != '\n') {
                        token_ = {loc(), tok::error, "expected new line after cariage return (bad line ending)"};
                        return;
                    }
                    continue; // catch the new line on the next pass

                // end of file
                case 0      :
                    token_ = {loc(), tok::eof, "eof"s};
                    return;

                case ';':
                    eat_comment();
                    continue;
                case '(':
                    token_ = {loc(), tok::lparen, {character()}};
                    return;
                case ')':
                    token_ = {loc(), tok::rparen, {character()}};
                    return;
                ARB_CASE_LETTERS
                    token_ = symbol();
                    return;
                ARB_CASE_DIGITS
                    token_ = number();
                    return;
                case '"':
                    token_ = string();
                    return;
                case '-':
                case '+':
                case '.':
                    {
                        if (empty()) {
                            token_ = {loc(), tok::error, "Unexpected end of input."};
                            return;
                        }
                        char c = peek(1);
                        if (std::isdigit(c) or c=='.') {
                            token_ = number();
                            return;
                        }
                    }
                    token_ = {loc(), tok::error,
                        util::pprintf("Unexpected character '{}'.", character())};
                    return;

                default:
                    token_ = {loc(), tok::error,
                        util::pprintf("Unexpected character '{}'.", character())};
                    return;
            }
        }
#undef ARB_CASE_LETTERS
#undef ARB_CASE_DIGITS

        if (!empty()) {
            // todo: handle error: should never hit this
        }
        token_ = {loc(), tok::eof, "eof"s};
        return;
    }

    // Look ahead n characters in the input stream.
    // If peek to or past the end of the stream return '\0'.
    char peek(int n) {
        const char* c = stream_;
        while (*c && n--) ++c;
        return *c;
    }

    // Consumes characters in the stream until end of stream or a new line.
    // Assumes that the current location is the `;` that starts the comment.
    void eat_comment() {
        while (!empty() && *stream_!='\n') {
            ++stream_;
        }
    }

    // Parse alphanumeric sequence that starts with an alphabet character,
    // and my contain alphabet, numeric or one of {+ -  *  /  @  $  %  ^  &  _  =  <  >  ~ .}
    //
    // This definition follows the symbol naming conventions of common lisp, without the
    // use of pipes || to define symbols with arbitrary strings.
    //
    // Valid symbols:
    //    sub_dendrite
    //    sub-dendrite
    //    sub-dendrite:
    //    foo@3.2/lower
    //    temp_
    //    branch3
    //    A
    // Invalid symbols:
    //    _cat          ; can't start with underscore
    //    -cat          ; can't start with hyphen
    //    2ndvar        ; can't start with numeric character
    //
    // Returns the appropriate token kind if symbol is a keyword.
    token symbol() {
        auto start = loc();
        std::string symbol;
        char c = *stream_;

        // Assert that current position is at the start of an identifier
        if( !(std::isalpha(c)) ) {
            throw s_expr_lexer_error(
                "Lexer attempting to read identifier when none is available", loc());
        }

        symbol += c;
        ++stream_;
        while(1) {
            c = *stream_;

            if(is_valid_symbol_char(c)) {
                symbol += c;
                ++stream_;
            }
            else {
                break;
            }
        }

        // test if the symbol matches a keyword
        auto it = keyword_to_tok.find(symbol);
        if (it!=keyword_to_tok.end()) {
            return {start, it->second, std::move(symbol)};
        }
        return {start, tok::symbol, std::move(symbol)};
    }

    token string() {
        using namespace std::string_literals;
        if (*stream_ != '"') {
            throw s_expr_lexer_error(
                "Lexer attempting to read string without opening \"", loc());
        }

        auto start = loc();
        ++stream_;
        std::string str;
        while (!empty() && *stream_!='"') {
            str.push_back(*stream_);
            ++stream_;
        }
        if (empty()) return {start, tok::error, "string missing closing \""};
        ++stream_; // gobble the closing "

        return {start, tok::string, str};
    }

    token number() {
        using namespace std::string_literals;

        auto start = loc();
        std::string str;
        char c = *stream_;

        // Start counting the number of points in the number.
        auto num_point = (c=='.' ? 1 : 0);
        auto uses_scientific_notation = 0;

        str += c;
        ++stream_;
        while(1) {
            c = *stream_;
            if (std::isdigit(c)) {
                str += c;
                ++stream_;
            }
            else if (c=='.') {
                if (++num_point>1) {
                    // Can't have more than one '.' in a number
                    return {start, tok::error, "unexpected '.'"s};
                }
                str += c;
                ++stream_;
                if (uses_scientific_notation) {
                    // Can't have a '.' in the mantissa
                    return {start, tok::error, "unexpected '.'"s};
                }
            }
            else if (!uses_scientific_notation && (c=='e' || c=='E')) {
                if ( std::isdigit(peek(1)) ||
                    (is_plusminus(peek(1)) && std::isdigit(peek(2))))
                {
                    uses_scientific_notation++;
                    str += c;
                    stream_++;
                    // Consume the next char if +/-
                    if (is_plusminus(*stream_)) {
                        str += *stream_++;
                    }
                }
                else {
                    // the 'e' or 'E' is the beginning of a new token
                    break;
                }
            }
            else {
                break;
            }
        }

        const bool is_real = uses_scientific_notation || num_point>0;
        return {start, (is_real? tok::real: tok::integer), std::move(str)};
    }

    char character() {
        return *stream_++;
    }
};

//
// s expression members
//

bool s_expr::is_atom() const {
    return state.index()==0;
}

const token& s_expr::atom() const {
    return std::get<0>(state);
}

const s_expr& s_expr::head() const {
    return std::get<1>(state).head.get();
}

const s_expr& s_expr::tail() const {
    return std::get<1>(state).tail.get();
}

s_expr& s_expr::head() {
    return std::get<1>(state).head.get();
}

s_expr& s_expr::tail() {
    return std::get<1>(state).tail.get();
}

s_expr::operator bool() const {
    return !(is_atom() && atom().kind==tok::nil);
}

// Assume that stream indented and ready to go at location to start printing.
std::ostream& print(std::ostream& o, const s_expr& x, int indent) {
    std::string in(std::string::size_type(2*indent), ' ');
    if (x.is_atom()) {
       return o << x.atom();
    }
    auto it = std::begin(x);
    auto end = std::end(x);
    bool first=true;
    o << "(";
    while (it!=end) {
        if (!first && !it->is_atom()) {
            o << "\n" << in;
            print(o, *it, indent+1);
            ++it;
            if (it!=end && it->is_atom()) {
                o << "\n" << in;
            }
        }
        else {
            print(o, *it, indent+1);
            if (++it!=end) {
                o << " ";
            }
        }
        first = false;
    }
    return o << ")";
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const s_expr& x) {
    return print(o, x, 1);
}

ARB_ARBOR_API std::size_t length(const s_expr& l) {
    // The length of an atom is 1.
    if (l.is_atom() && l) {
        return 1;
    }
    // nil marks the end of a list.
    if (!l) {
        return 0u;
    }
    return 1+length(l.tail());
}

ARB_ARBOR_API src_location location(const s_expr& l) {
    if (l.is_atom()) return l.atom().loc;
    return location(l.head());
}

//
// parsing s expressions
//

namespace impl {

// If there is a parsing error, then an atom with kind==tok::error is returned
// with the error string in its spelling.
s_expr parse(lexer& L) {
    using namespace std::string_literals;

    s_expr node;
    auto t = L.current();

    if (t.kind==tok::lparen) {
        t = L.next();
        s_expr* n = &node;
        while (true) {
            if (t.kind == tok::eof) {
                return token{t.loc, tok::error,
                    "Unexpected end of input. Missing a closing parenthesis ')'."};
            }
            if (t.kind == tok::error) {
                return t;
            }
            else if (t.kind == tok::rparen) {
                *n = token{t.loc, tok::nil, "nil"};
                t = L.next();
                break;
            }
            else if (t.kind == tok::lparen) {
                auto e = parse(L);
                if (e.is_atom() && e.atom().kind==tok::error) return e;
                *n = {std::move(e), {}};
                t = L.current();
            }
            else {
                *n = {s_expr(t), {}};
                t = L.next();
            }

            n = &n->tail();
        }
    }
    else if (t.kind==tok::eof) {
        return token{t.loc, tok::error, "Empty expression."};
    }
    else if (t.kind==tok::rparen) {
        return token{t.loc, tok::error, "Missing opening parenthesis'('."};
    }
    // an atom or an error
    else {
        L.next(); // advance the lexer to the next token
        return t;
    }

    return node;
}

}

ARB_ARBOR_API s_expr parse_s_expr(const std::string& line) {
    lexer l(line.c_str());
    s_expr result = impl::parse(l);
    const bool err = result.is_atom()? result.atom().kind==tok::error: false;
    if (!err) {
        auto t = l.current();
        if (t.kind!=tok::eof) {
            return token{t.loc, tok::error,
                         util::pprintf("Unexpected '{}' at the end of input.", t)};
        }
    }
    return result;
}
} // namespace arb
