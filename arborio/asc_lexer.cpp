#include <cctype>
#include <string>

#include <arborio/neurolucida.hpp>

#include "asc_lexer.hpp"

namespace arborio {

namespace asc {

std::ostream& operator<<(std::ostream& o, const tok& t) {
    switch (t) {
        case tok::lparen:
            return o << "lparen";
        case tok::rparen:
            return o << "rparen";
        case tok::lt:
            return o << "lt";
        case tok::gt:
            return o << "gt";
        case tok::comma:
            return o << "comma";
        case tok::real:
            return o << "real";
        case tok::integer:
            return o << "integer";
        case tok::symbol:
            return o << "symbol";
        case tok::string:
            return o << "string";
        case tok::pipe:
            return o << "pipe";
        case tok::eof:
            return o << "eof";
        case tok::error:
            return o << "error";
    }
    return o << "unknown";
}

std::ostream& operator<<(std::ostream& o, const src_location& l) {
    return o << "(src-location " << l.line << " " << l.column << ")";
}

std::ostream& operator<<(std::ostream& o, const token& t) {
    const char* spelling = t.spelling.c_str();
    if      (t.kind==tok::eof)   spelling = "\\0";
    else if (t.kind==tok::error) spelling = "";
    return o << "(token " << t.kind << " \"" << spelling << "\" " << t.loc << ")";
}

inline bool is_alphanumeric(char c) {
    return std::isalnum(static_cast<unsigned char>(c));
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

class lexer_impl {
    const char* line_start_;
    const char* stream_;
    unsigned line_;
    token token_;

public:

    lexer_impl(const char* begin):
        line_start_(begin), stream_(begin), line_(0)
    {
        // Prime the first token.
        parse();
    }

    // Return the current token in the stream.
    const token& current() {
        return token_;
    }

    const token& next(unsigned n=1) {
        while (n--) parse();
        return token_;
    }

    token peek(unsigned n) {
        // Save state.
        auto ls = line_start_;
        auto st = stream_;
        auto l  = line_;
        auto t  = token_;

        // Advance n tokens.
        next(n);

        // Restore state.
        std::swap(t, token_);
        line_ = l;
        line_start_ = ls;
        stream_ = st;

        return t;
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
                case '<':
                    token_ = {loc(), tok::lt, "<"};
                    ++stream_;
                    return;
                case '>':
                    token_ = {loc(), tok::gt, ">"};
                    ++stream_;
                    return;
                case ',':
                    token_ = {loc(), tok::comma, ","};
                    ++stream_;
                    return;
                case '|':
                    token_ = {loc(), tok::pipe, "|"};
                    ++stream_;
                    return;
                case '-':
                case '+':
                case '.':
                    {
                        if (empty()) {
                            token_ = {loc(), tok::error, "Unexpected end of input."};
                            return;
                        }
                        char c = peek_char(1);
                        if (std::isdigit(c) or c=='.') {
                            token_ = number();
                            return;
                        }
                    }
                    token_ = {loc(), tok::error, std::string("Unexpected character '")+character()+"'"};
                    return;

                default:
                    token_ = {loc(), tok::error, std::string("Unexpected character '")+character()+"'"};
                    return;
            }
        }
#undef ARB_CASE_LETTERS
#undef ARB_CASE_DIGITS

        if (!empty()) {
            token_ = {loc(), tok::error, "Internal lexer error: expected end of input, please open a bug report"s};
            return;
        }
        token_ = {loc(), tok::eof, "eof"s};
        return;
    }

    // Look ahead n characters in the input stream.
    // If peek to or past the end of the stream return '\0'.
    char peek_char(int n) {
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
        using namespace std::string_literals;
        auto start = loc();
        std::string symbol;
        char c = *stream_;

        // Assert that current position is at the start of an identifier
        if( !(std::isalpha(c)) ) {
            return {start, tok::error, "Internal error: lexer attempting to read identifier when none is available '.'"s};
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

        return {start, tok::symbol, std::move(symbol)};
    }

    token string() {
        using namespace std::string_literals;
        if (*stream_ != '"') {
            return {loc(), tok::error, "Internal error: lexer attempting to read identifier when none is available '.'"s};
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
                if ( std::isdigit(peek_char(1)) ||
                    (is_plusminus(peek_char(1)) && std::isdigit(peek_char(2))))
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

lexer::lexer(const char* begin):
    impl_(new lexer_impl(begin))
{}

const token& lexer::current() {
    return impl_->current();
}

const token& lexer::next(unsigned n) {
    return impl_->next(n);
}

token lexer::peek(unsigned n) {
    return impl_->peek(n);
}

lexer::~lexer() = default;

} // namespace asc

} // namespace arborio
