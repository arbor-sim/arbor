#include <cstdio>

#include <iostream>
#include <string>

#include "lexer.hpp"
#include "io/pprintf.hpp"

// helpers for identifying character types
inline bool in_range(char c, char first, char last) {
    return c>=first && c<=last;
}
inline bool is_numeric(char c) {
    return in_range(c, '0', '9');
}
inline bool is_alpha(char c) {
    return (in_range(c, 'a', 'z') || in_range(c, 'A', 'Z') );
}
inline bool is_alphanumeric(char c) {
    return (is_numeric(c) || is_alpha(c) );
}
inline bool is_eof(char c) {
    return (c==0);
}
inline bool is_operator(char c) {
    return (c=='+' || c=='-' || c=='*' || c=='/' || c=='^' || c=='\'');
}
inline bool is_plusminus(char c) {
    return (c=='+' || c=='-');
}

//*********************
// Lexer
//*********************

Token Lexer::parse() {
    Token t;

    // the while loop strips white space/new lines in front of the next token
    while(1) {
        location_.column = current_-line_+1;
        t.location = location_;

        switch(*current_) {
            // end of file
            case 0      :       // end of string
                t.spelling = "eof";
                t.type = tok::eof;
                return t;

            // white space
            case ' '    :
            case '\t'   :
            case '\v'   :
            case '\f'   :
                current_++;
                continue;   // skip to next character

            // new line
            case '\n'   :
                current_++;
                line_ = current_;
                location_.line++;
                continue;   // skip to next line

            // new line
            case '\r'   :
                current_++;
                if(*current_ != '\n') {
                    error_string_ = pprintf("bad line ending: \\n must follow \\r");
                    status_ = lexerStatus::error;
                    t.type = tok::reserved;
                    return t;
                }
                current_++;
                line_ = current_;
                location_.line++;
                continue;   // skip to next line

            // comment (everything after : or ? on a line is a comment)
            case ':' :
            case '?' :
                // strip characters until either end of file or end of line
                while( !is_eof(*current_) && *current_ != '\n') {
                    ++current_;
                }
                continue;

            // number
            case '0' ... '9':
            case '.':
                return number();

            // identifier or keyword
            case 'a' ... 'z':
            case 'A' ... 'Z':
            case '_': {
                // get std::string of the identifier
                auto id = identifier();
                if (id == "UNITSON" || id == "UNITSOFF") continue;
                if (id == "COMMENT") {
                    while (!is_eof(*current_)) {
                        while ((*current_ != '\n') && (*current_ != '\r') && !is_alpha(*current_)) {
                            current_++;
                        }
                        if (*current_ == '\n') {
                            current_++;
                            line_ = current_;
                            location_.line++;
                        }
                        else if (*current_ == '\r') {
                            current_++;
                            if(*current_ != '\n') {
                                error_string_ = pprintf("bad line ending: \\n must follow \\r");
                                return t;
                            }
                            current_++;
                            line_ = current_;
                            location_.line++;
                        }
                        else if (identifier() == "ENDCOMMENT") break;
                    }
                    continue;
                }
                t.spelling = id;
                t.type = status_ == lexerStatus::error
                          ? tok::reserved
                          : get_identifier_type(t.spelling);
                return t;
            }
            case '(':
                t.type = tok::lparen;
                t.spelling += character();
                return t;
            case ')':
                t.type = tok::rparen;
                t.spelling += character();
                return t;
            case '{':
                t.type = tok::lbrace;
                t.spelling += character();
                return t;
            case '}':
                t.type = tok::rbrace;
                t.spelling += character();
                return t;
            case '~':
                t.type = tok::tilde;
                t.spelling += character();
                return t;
            case '=': {
                t.spelling += character();
                if(*current_=='=') {
                    t.spelling += character();
                    t.type=tok::equality;
                }
                else {
                    t.type = tok::eq;
                }
                return t;
            }
            case '!': {
                t.spelling += character();
                if(*current_=='=') {
                    t.spelling += character();
                    t.type=tok::ne;
                }
                else {
                    t.type = tok::lnot;
                }
                return t;
            }
            case '+':
                t.type = tok::plus;
                t.spelling += character();
                return t;
            case '-':
                t.type = tok::minus;
                t.spelling += character();
                return t;
            case '/':
                t.type = tok::divide;
                t.spelling += character();
                return t;
            case '*':
                t.type = tok::times;
                t.spelling += character();
                return t;
            case '^':
                t.type = tok::pow;
                t.spelling += character();
                return t;
            // comparison binary operators
            case '<': {
                t.spelling += character();
                if(*current_=='=') {
                    t.spelling += character();
                    t.type = tok::lte;
                }
                else if(*current_=='-' && current_[1]=='>') {
                    t.spelling += character();
                    t.spelling += character();
                    t.type = tok::arrow;
                }
                else {
                    t.type = tok::lt;
                }
                return t;
            }
            case '>': {
                t.spelling += character();
                if(*current_=='=') {
                    t.spelling += character();
                    t.type = tok::gte;
                }
                else {
                    t.type = tok::gt;
                }
                return t;
            }
            case '&': {
                bool valid = false;
                t.spelling += character();
                if (*current_ == '&') {
                    t.spelling += character();
                    if (*current_ != '&') {
                        t.type = tok::land;
                        valid = true;
                    }
                }
                if (!valid) {
                    error_string_ = pprintf("& must be in pairs");
                    status_ = lexerStatus::error;
                    t.type = tok::reserved;
                }
                return t;
            }
            case '|': {
                bool valid = false;
                t.spelling += character();
                if (*current_ == '|') {
                    t.spelling += character();
                    if (*current_ != '|') {
                        t.type = tok::lor;
                        valid = true;
                    }
                }
                if (!valid) {
                    error_string_ = pprintf("| must be in pairs");
                    status_ = lexerStatus::error;
                    t.type = tok::reserved;
                }
                return t;
            }
            case '\'':
                t.type = tok::prime;
                t.spelling += character();
                return t;
            case ',':
                t.type = tok::comma;
                t.spelling += character();
                return t;
            default:
                error_string_ =
                    pprintf( "unexpected character '%' at %",
                             *current_, location_);
                status_ = lexerStatus::error;
                t.spelling += character();
                t.type = tok::reserved;
                return t;
        }
    }

    // return the token
    return t;
}

Token Lexer::peek() {
    // save the current position
    const char *oldpos  = current_;
    const char *oldlin  = line_;
    Location    oldloc  = location_;

    Token t = parse(); // read the next token

    // reset position
    current_  = oldpos;
    location_ = oldloc;
    line_     = oldlin;

    return t;
}

bool Lexer::search_to_eol(tok const& t) {
    // save the current position
    const char *oldpos  = current_;
    const char *oldlin  = line_;
    Location    oldloc  = location_;

    Token p = token_;
    bool ret = false;
    while (line_ == oldlin && p.type != tok::eof) {
        if (p.type == t) {
            ret = true;
            break;
        }
        p = parse();
    }

    // reset position
    current_  = oldpos;
    location_ = oldloc;
    line_     = oldlin;

    return ret;
}

// scan floating point number from stream
Token Lexer::number() {
    std::string str;
    char c = *current_;

    // start counting the number of points in the number
    auto num_point = (c=='.' ? 1 : 0);
    auto uses_scientific_notation = 0;
    bool incorrectly_formed_mantisa = false;

    str += c;
    current_++;
    while(1) {
        c = *current_;
        if(is_numeric(c)) {
            str += c;
            current_++;
        }
        else if(c=='.') {
            num_point++;
            str += c;
            current_++;
            if(uses_scientific_notation) {
                incorrectly_formed_mantisa = true;
            }
        }
        else if(!uses_scientific_notation && (c=='e' || c=='E')) {
            if(is_numeric(current_[1]) ||
               (is_plusminus(current_[1]) && is_numeric(current_[2])))
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

    // check that the mantisa is an integer
    if(incorrectly_formed_mantisa) {
        error_string_ = pprintf("the exponent/mantissa must be an integer '%'", yellow(str));
        status_ = lexerStatus::error;
    }
    // check that there is at most one decimal point
    // i.e. disallow values like 2.2324.323
    if(num_point>1) {
        error_string_ = pprintf("too many .'s when reading the number '%'", yellow(str));
        status_ = lexerStatus::error;
    }

    tok type;
    if(status_==lexerStatus::error) {
        type = tok::reserved;
    }
    else if(num_point<1 && !uses_scientific_notation) {
        type = tok::integer;
    }
    else {
        type = tok::real;
    }

    return Token(type, str, location_);
}

// scan identifier from stream
//  examples of valid names:
//      _1 _a ndfs var9 num_a This_
//  examples of invalid names:
//      _ __ 9val 9_
std::string Lexer::identifier() {
    std::string name;
    char c = *current_;

    // assert that current position is at the start of a number
    // note that the first character can't be numeric
    if( !(is_alpha(c) || c=='_') ) {
        throw compiler_exception(
            "Lexer attempting to read number when none is available",
            location_);
    }

    name += c;
    current_++;
    while(1) {
        c = *current_;

        if(is_alphanumeric(c) || c=='_') {
            name += c;
            current_++;
        }
        else {
            break;
        }
    }

    return name;
}

// scan a single character from the buffer
char Lexer::character() {
    return *current_++;
}

std::map<tok, int> Lexer::binop_prec_;

void Lexer::binop_prec_init() {
    if(binop_prec_.size()>0)
        return;

    // I have taken the operator precedence from C++
    // Note that only infix operators require precedence.
    binop_prec_[tok::eq]       = 1;
    binop_prec_[tok::lor]      = 2;
    binop_prec_[tok::land]     = 3;
    binop_prec_[tok::equality] = 4;
    binop_prec_[tok::ne]       = 4;
    binop_prec_[tok::lt]       = 5;
    binop_prec_[tok::lte]      = 5;
    binop_prec_[tok::gt]       = 5;
    binop_prec_[tok::gte]      = 5;
    binop_prec_[tok::plus]     = 6;
    binop_prec_[tok::minus]    = 6;
    binop_prec_[tok::times]    = 7;
    binop_prec_[tok::divide]   = 7;
    binop_prec_[tok::pow]      = 8;
}

int Lexer::binop_precedence(tok tok) {
    auto r = binop_prec_.find(tok);
    if(r==binop_prec_.end())
        return -1;
    return r->second;
}

associativityKind Lexer::operator_associativity(tok token) {
    if(token==tok::pow) {
        return associativityKind::right;
    }
    return associativityKind::left;
}

// pre  : identifier is a valid identifier ([_a-zA-Z][_a-zA-Z0-9]*)
// post : if(identifier is a keyword) return tok::<keyword>
//        else                        return tok::identifier
tok Lexer::get_identifier_type(std::string const& identifier) {
    auto pos = keyword_map.find(identifier);
    return pos==keyword_map.end() ? tok::identifier : pos->second;
}
