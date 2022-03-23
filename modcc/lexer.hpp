#pragma once

// inspiration was taken from the Digital Mars D compiler
//      github.com/D-Programming-Language/dmd

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "location.hpp"
#include "error.hpp"
#include "token.hpp"
#include <libmodcc/export.hpp>

// status of the lexer
enum class lexerStatus {
    error,  // lexer has encounterd a problem
    happy   // lexer is in a good place
};

// associativity of an operator
enum class associativityKind {
    left,
    right
};

bool is_keyword(Token const& t);


// class that implements the lexer
// takes a range of characters as input parameters
class ARB_LIBMODCC_API Lexer {
public:
    Lexer(const char* begin, const char* end)
    :   begin_(begin),
        end_(end),
        current_(begin),
        line_(begin),
        location_()
    {
        if(begin_>end_) {
            throw std::out_of_range("Lexer(begin, end) : begin>end");
        }

        initialize_token_maps();
        binop_prec_init();
    }

    Lexer(std::vector<char> const& v)
    :   Lexer(v.data(), v.data()+v.size())
    {}

    Lexer(std::string const& s)
    :   buffer_(s.data(), s.data()+s.size()+1)
    {
        begin_   = buffer_.data();
        end_     = buffer_.data() + buffer_.size();
        current_ = begin_;
        line_    = begin_;

        initialize_token_maps();
        binop_prec_init();
    }

    // get the next token
    Token parse();

    void get_token() {
        token_ = parse();
    }

    // return the next token in the stream without advancing the current position
    Token peek();

    // Look for `t` until new line or eof without advancing the current position, return true if found
    bool search_to_eol(tok const& t);

    // scan a number from the stream
    Token number();

    // scan an identifier string from the stream
    std::string identifier();

    // scan a character from the stream
    char character();

    Location location() {return location_;}

    // binary operator precedence
    static std::map<tok, int> binop_prec_;

    lexerStatus status() {return status_;}

    const std::string& error_message() {return error_string_;};

    static int binop_precedence(tok tok);
    static associativityKind operator_associativity(tok token);
protected:
    // buffer used for short-lived parsers
    std::vector<char> buffer_;

    // generate lookup tables (hash maps) for keywords
    void keywords_init();
    void token_strings_init();
    void binop_prec_init();

    // helper for determining if an identifier string matches a keyword
    tok get_identifier_type(std::string const& identifier);

    const char *begin_, *end_;// pointer to start and 1 past the end of the buffer
    const char *current_;     // pointer to current character
    const char *line_;        // pointer to start of current line
    Location location_;  // current location (line,column) in buffer

    lexerStatus status_ = lexerStatus::happy;
    std::string error_string_;

    Token token_;
};
