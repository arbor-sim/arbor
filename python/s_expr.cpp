#include <iostream>

#include <cctype>
#include <cstring>
#include <string>
#include <memory>
#include <ostream>
#include <vector>

#include <arbor/util/either.hpp>
#include <arbor/arbexcept.hpp>

#include "s_expr.hpp"
#include "strprintf.hpp"

namespace pyarb {

inline bool is_alphanumeric(char c) {
    return std::isdigit(c) || std::isalpha(c);
}
inline bool is_plusminus(char c) {
    return (c=='-' || c=='+');
}

std::ostream& operator<<(std::ostream& o, const tok& t) {
    switch (t) {
        case tok::nil:    return o << "nil";
        case tok::lparen: return o << "lparen";
        case tok::rparen: return o << "rparen";
        case tok::real:   return o << "real";
        case tok::integer:return o << "integer";
        case tok::name:   return o << "name";
        case tok::string: return o << "string";
        case tok::eof:    return o << "eof";
        case tok::error:  return o << "error";
    }
    return o << "<unknown>";
}

std::ostream& operator<<(std::ostream& o, const token& t) {
    return o << util::pprintf("{}", t.spelling);
}

//
// lexer
//

struct parser_error: public arb::arbor_exception {
    int loc;
    parser_error(const std::string& msg, int l):
        arbor_exception(msg), loc(l)
    {}
};

static std::unordered_map<tok, const char*> tok_to_keyword = {
    {tok::nil,    "nil"},
};

static std::unordered_map<std::string, tok> keyword_to_tok = {
    {"nil",    tok::nil},
};

class lexer {
    const char* data_;;
    const char* end_;;
    const char* current_;
    int loc_;
    token token_;

public:

    lexer(const char* s):
        data_(s),
        end_(data_ + std::strlen(data_)),
        current_(data_)
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

    // Consume the and return the next token in the stream.
    void parse() {
        using namespace std::string_literals;

        while (current_!=end_) {
            loc_ = current_-data_;
            switch (*current_) {
                // end of file
                case 0      :       // end of string
                    token_ = {loc_, tok::eof, "eof"s};
                    return;

                // white space
                case ' '    :
                case '\t'   :
                case '\v'   :
                case '\f'   :
                case '\n'   :
                    character();
                    continue;   // skip to next character

                case '(':
                    token_ = {loc_, tok::lparen, {character()}};
                    return;
                case ')':
                    token_ = {loc_, tok::rparen, {character()}};
                    return;
                case 'a' ... 'z':
                case 'A' ... 'Z':
                    token_ = name();
                    return;
                case '0' ... '9':
                    token_ = number();
                    return;
                case '"':
                    token_ = string();
                    return;
                case '-':
                case '+':
                    {
                        char c = current_[1];
                        if (std::isdigit(c) or c=='.') {
                            token_ = number();
                            return;
                        }
                    }
                    token_ = {loc_, tok::error,
                        util::pprintf("Unexpected character '{}'.", character())};
                    return;

                default:
                    token_ = {loc_, tok::error,
                        util::pprintf("Unexpected character '{}'.", character())};
                    return;
            }
        }

        if (current_!=end_) {
            // todo: handle error: should never hit this
        }
        token_ = {loc_, tok::eof, "eof"s};
        return;
    }

    // Parse alphanumeric sequence that starts with an alphabet character,
    // and my contain alphabet, numeric or underscor '_' characters.
    //
    // Valid names:
    //    sub_dendrite
    //    temp_
    //    branch3
    //    A
    // Invalid names:
    //    _cat          ; can't start with underscore
    //    2ndvar        ; can't start with numeric character
    //
    // Returns the appropriate token kind if name is a keyword.
    token name() {
        std::string name;
        char c = *current_;

        // Assert that current position is at the start of an identifier
        if( !(std::isalpha(c)) ) {
            throw parser_error(
                "Lexer attempting to read identifier when none is available", loc_);
        }

        name += c;
        ++current_;
        while(1) {
            c = *current_;

            if(is_alphanumeric(c) || c=='_') {
                name += character();
            }
            else {
                break;
            }
        }

        // test if the name matches a keyword
        auto it = keyword_to_tok.find(name.c_str());
        if (it!=keyword_to_tok.end()) {
            return {loc_, it->second, std::move(name)};
        }
        return {loc_, tok::name, std::move(name)};
    }

    token string() {
        using namespace std::string_literals;

        ++current_;
        const char* begin = current_;
        while (current_!=end_ && character()!='"');

        if (current_==end_) return {loc_, tok::error, "string missing closing \""};

        return {loc_, tok::string, std::string(begin, current_-1)};
    }

    token number() {
        using namespace std::string_literals;

        std::string str;
        char c = *current_;

        // Start counting the number of points in the number.
        auto num_point = (c=='.' ? 1 : 0);
        auto uses_scientific_notation = 0;

        str += c;
        current_++;
        while(1) {
            c = *current_;
            if(std::isdigit(c)) {
                str += c;
                current_++;
            }
            else if(c=='.') {
                if (++num_point>1) {
                    // Can't have more than one '.' in a number
                    return {int(current_-data_), tok::error, "unexpected '.'"s};
                }
                str += c;
                current_++;
                if(uses_scientific_notation) {
                    // Can't have a '.' in the mantissa
                    return {int(current_-data_), tok::error, "unexpected '.'"s};
                }
            }
            else if(!uses_scientific_notation && (c=='e' || c=='E')) {
                if(std::isdigit(current_[1]) ||
                   (is_plusminus(current_[1]) && std::isdigit(current_[2])))
                {
                    uses_scientific_notation++;
                    str += c;
                    current_++;
                    // Consume the next char if +/-
                    if (is_plusminus(*current_)) {
                        str += *current_++;
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
        return {loc_, (is_real? tok::real: tok::integer), std::move(str)};
    }

    char character() {
        return *current_++;
    }
};

bool test_identifier(const char* in) {
    lexer L(in);
    auto x = L.current();
    return x.kind==tok::name && x.spelling==in;
}

//
// s expression members
//

bool s_expr::is_atom() const {
    return (bool)state;
}

const token& s_expr::atom() const {
    return state.get<0>();
}

const s_expr& s_expr::head() const {
    return state.get<1>().head.get();
}

const s_expr& s_expr::tail() const {
    return state.get<1>().tail.get();
}

s_expr& s_expr::head() {
    return state.get<1>().head.get();
}

s_expr& s_expr::tail() {
    return state.get<1>().tail.get();
}

s_expr::operator bool() const {
    return !(is_atom() && atom().kind==tok::nil);
}

std::ostream& operator<<(std::ostream& o, const s_expr& x) {
    if (x.is_atom()) return o << x.atom();
#if 0 // print full tree with terminating 'nil'
    return o << "(" << x.head() << " . " << x.tail() << ")";
#else // print '(a . nil)' as 'a'
    return x.tail()? o << "(" << x.head() << " . " << x.tail() << ")"
                   : o << x.head();
#endif
}

std::size_t length(const s_expr& l) {
    if (l.is_atom() && l) {
        throw arb::arbor_internal_error(
            util::pprintf("Internal error: can't take length of an atom in '{}'.", l));
    }
    if (!l) { // nil
        return 0u;
    }
    return 1+length(l.tail());
}

int location(const s_expr& l) {
    if (l.is_atom()) return l.atom().column;
    return location(l.head());
}

//
// parsing s expressions
//

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
                return token{t.column, tok::error,
                    "Unexpected end of input. Missing a closing parenthesis ')'."};;
            }
            if (t.kind == tok::error) {
                return t;
            }
            else if (t.kind == tok::rparen) {
                *n = token{t.column, tok::nil, "nil"};
                break;
            }
            else if (t.kind == tok::lparen) {
                auto e = parse(L);
                if (e.is_atom() && e.atom().kind==tok::error) return e;
                *n = {std::move(e), {}};
            }
            else {
                *n = {s_expr(t), {}};
            }

            n = &n->tail();
            t = L.next();
        }
    }
    else {
        return token{t.column, tok::error, "Missing opening parenthesis'('."};;
    }

    return node;
}

s_expr parse(const char* in) {
    lexer l(in);
    s_expr result = parse(l);
    const bool err = result.is_atom()? result.atom().kind==tok::error: false;
    if (!err) {
        auto t = l.next(); // pop the last rparen token.
        if (t.kind!=tok::eof) {
            return token{t.column, tok::error,
                         util::pprintf("Unexpected '{}' at the end of input.", t)};
        }
    }
    return result;
}

s_expr parse(const std::string& in) {
    return parse(in.c_str());
}

} // namespace pyarb

