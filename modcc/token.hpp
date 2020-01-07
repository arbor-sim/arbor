#pragma once

#include <string>
#include <map>
#include <unordered_map>

#include "location.hpp"

enum class tok {
    eof, // end of file

    /////////////////////////////
    // symbols
    /////////////////////////////

    // infix binary ops

    // = + - * / ^
    eq, plus, minus, times, divide, pow, land, lor,
    // comparison
    lnot,    // !   named logical not, to avoid clash with C++ not keyword
    lt,      // <
    lte,     // <=
    gt,      // >
    gte,     // >=
    equality,// ==
    ne,      // !=

    // <->
    arrow,

    // ~
    tilde,

    // , '
    comma, prime,

    // { }
    lbrace, rbrace,
    // ( )
    lparen, rparen,

    // variable/function names
    identifier,

    // numbers
    real, integer,

    /////////////////////////////
    // keywords
    /////////////////////////////
    // block keywoards
    title,
    neuron, units, parameter,
    constant, assigned, state, breakpoint,
    derivative, kinetic, procedure, initial, function, linear,
    net_receive,

    // keywoards inside blocks
    unitsoff, unitson,
    suffix, nonspecific_current, useion,
    read, write, valence,
    range, local, conserve, compartment,
    solve, method, steadystate,
    threadsafe, global,
    point_process,

    // prefix binary operators
    min, max,

    // unary operators
    exp, sin, cos, log, abs, safeinv,
    exprelr, // equivalent to x/(exp(x)-1) with exprelr(0)=1

    // logical keywords
    if_stmt, else_stmt, // add _stmt to avoid clash with c++ keywords

    // solver methods
    cnexp,
    sparse,

    conductance,

    reserved, // placeholder for generating keyword lookup
};

// what is in a token?
//  tok indicating type of token
//  information about its location
struct Token {
    // the spelling string contains the text of the token as it was written
    // in the input file
    //   type = tok::real       : spelling = "3.1415"  (e.g.)
    //   type = tok::identifier : spelling = "foo_bar" (e.g.)
    //   type = tok::plus       : spelling = "+"       (always)
    //   type = tok::if         : spelling = "if"      (always)
    std::string spelling;
    tok type;
    Location location;

    Token(tok tok, std::string const& sp, Location loc=Location(0,0))
    :   spelling(sp),
        type(tok),
        location(loc)
    {}

    Token()
    :   spelling(""),
        type(tok::reserved),
        location(Location())
    {};
};

// lookup table used for checking if an identifier matches a keyword
extern std::unordered_map<std::string, tok> keyword_map;

// for stringifying a token type
extern std::map<tok, std::string> token_map;

void initialize_token_maps();
std::string token_string(tok token);
bool is_keyword(Token const& t);
std::ostream& operator<< (std::ostream& os, Token const& t);

