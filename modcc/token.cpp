#include <mutex>

#include "token.hpp"

// lookup table used for checking if an identifier matches a keyword
std::unordered_map<std::string, tok> keyword_map;

// for stringifying a token type
std::map<tok, std::string> token_map;

std::mutex mutex;

struct Keyword {
    const char *name;
    tok type;
};

struct TokenString {
    const char *name;
    tok token;
};

static Keyword keywords[] = {
    {"TITLE",       tok::title},
    {"NEURON",      tok::neuron},
    {"UNITS",       tok::units},
    {"PARAMETER",   tok::parameter},
    {"CONSTANT",    tok::constant},
    {"ASSIGNED",    tok::assigned},
    {"STATE",       tok::state},
    {"BREAKPOINT",  tok::breakpoint},
    {"DERIVATIVE",  tok::derivative},
    {"KINETIC",     tok::kinetic},
    {"LINEAR",      tok::linear},
    {"PROCEDURE",   tok::procedure},
    {"FUNCTION",    tok::function},
    {"INITIAL",     tok::initial},
    {"NET_RECEIVE", tok::net_receive},
    {"UNITSOFF",    tok::unitsoff},
    {"UNITSON",     tok::unitson},
    {"SUFFIX",      tok::suffix},
    {"NONSPECIFIC_CURRENT", tok::nonspecific_current},
    {"USEION",      tok::useion},
    {"READ",        tok::read},
    {"WRITE",       tok::write},
    {"VALENCE",     tok::valence},
    {"RANGE",       tok::range},
    {"LOCAL",       tok::local},
    {"CONSERVE",    tok::conserve},
    {"SOLVE",       tok::solve},
    {"THREADSAFE",  tok::threadsafe},
    {"GLOBAL",      tok::global},
    {"POINT_PROCESS", tok::point_process},
    {"COMPARTMENT", tok::compartment},
    {"METHOD",      tok::method},
    {"STEADYSTATE", tok::steadystate},
    {"if",          tok::if_stmt},
    {"else",        tok::else_stmt},
    {"cnexp",       tok::cnexp},
    {"sparse",      tok::sparse},
    {"min",         tok::min},
    {"max",         tok::max},
    {"exp",         tok::exp},
    {"sin",         tok::sin},
    {"cos",         tok::cos},
    {"log",         tok::log},
    {"fabs",        tok::abs},
    {"exprelr",     tok::exprelr},
    {"safeinv",     tok::safeinv},
    {"CONDUCTANCE", tok::conductance},
    {nullptr,       tok::reserved},
};

static TokenString token_strings[] = {
    {"=",           tok::eq},
    {"+",           tok::plus},
    {"-",           tok::minus},
    {"*",           tok::times},
    {"/",           tok::divide},
    {"^",           tok::pow},
    {"!",           tok::lnot},
    {"<",           tok::lt},
    {"<=",          tok::lte},
    {">",           tok::gt},
    {">=",          tok::gte},
    {"==",          tok::equality},
    {"!=",          tok::ne},
    {"&&",          tok::land},
    {"||",          tok::lor},
    {"<->",         tok::arrow},
    {"~",           tok::tilde},
    {",",           tok::comma},
    {"'",           tok::prime},
    {"{",           tok::lbrace},
    {"}",           tok::rbrace},
    {"(",           tok::lparen},
    {")",           tok::rparen},
    {"identifier",  tok::identifier},
    {"real",        tok::real},
    {"integer",     tok::integer},
    {"TITLE",       tok::title},
    {"NEURON",      tok::neuron},
    {"UNITS",       tok::units},
    {"PARAMETER",   tok::parameter},
    {"CONSTANT",    tok::constant},
    {"ASSIGNED",    tok::assigned},
    {"STATE",       tok::state},
    {"BREAKPOINT",  tok::breakpoint},
    {"DERIVATIVE",  tok::derivative},
    {"KINETIC",     tok::kinetic},
    {"LINEAR",      tok::linear},
    {"PROCEDURE",   tok::procedure},
    {"FUNCTION",    tok::function},
    {"INITIAL",     tok::initial},
    {"NET_RECEIVE", tok::net_receive},
    {"UNITSOFF",    tok::unitsoff},
    {"UNITSON",     tok::unitson},
    {"SUFFIX",      tok::suffix},
    {"NONSPECIFIC_CURRENT", tok::nonspecific_current},
    {"USEION",      tok::useion},
    {"READ",        tok::read},
    {"WRITE",       tok::write},
    {"VALENCE",     tok::valence},
    {"RANGE",       tok::range},
    {"LOCAL",       tok::local},
    {"SOLVE",       tok::solve},
    {"THREADSAFE",  tok::threadsafe},
    {"GLOBAL",      tok::global},
    {"POINT_PROCESS", tok::point_process},
    {"COMPARTMENT", tok::compartment},
    {"METHOD",      tok::method},
    {"STEADYSTATE", tok::steadystate},
    {"if",          tok::if_stmt},
    {"else",        tok::else_stmt},
    {"eof",         tok::eof},
    {"min",         tok::min},
    {"max",         tok::max},
    {"exp",         tok::exp},
    {"log",         tok::log},
    {"fabs",        tok::abs},
    {"exprelr",     tok::exprelr},
    {"safeinv",     tok::safeinv},
    {"cos",         tok::cos},
    {"sin",         tok::sin},
    {"cnexp",       tok::cnexp},
    {"CONDUCTANCE", tok::conductance},
    {"error",       tok::reserved},
};

/// set up lookup tables for converting between tokens and their
/// string representations
void initialize_token_maps() {
    // ensure that tables are initialized only once
    std::lock_guard<std::mutex> g(mutex);

    if(keyword_map.size()==0) {
        //////////////////////
        /// keyword map
        //////////////////////
        for(int i = 0; keywords[i].name!=nullptr; ++i) {
            keyword_map.insert( {keywords[i].name, keywords[i].type} );
        }

        //////////////////////
        // token map
        //////////////////////
        int i;
        for(i = 0; token_strings[i].token!=tok::reserved; ++i) {
            token_map.insert( {token_strings[i].token, token_strings[i].name} );
        }
        // insert the last token: tok::reserved
        token_map.insert( {token_strings[i].token, token_strings[i].name} );
    }
}

std::string token_string(tok token) {
    auto pos = token_map.find(token);
    return pos==token_map.end() ? std::string("<unknown token>") : pos->second;
}

bool is_keyword(Token const& t) {
    for(Keyword *k=keywords; k->name!=nullptr; ++k)
        if(t.type == k->type)
            return true;
    return false;
}

std::ostream& operator<< (std::ostream& os, Token const& t) {
    return os << "<<" << token_string(t.type) << ", " << t.spelling << ", " << t.location << ">>";
}
