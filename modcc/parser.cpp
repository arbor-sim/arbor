#include <cstring>
#include <string>

#include "parser.hpp"
#include "perfvisitor.hpp"
#include "token.hpp"
#include "util.hpp"

#include "io/pprintf.hpp"

// specialize on const char* for lazy evaluation of compile time strings
bool Parser::expect(tok tok, const char* str) {
    if (tok == token_.type) {
        return true;
    }

    error(
        strlen(str) > 0 ? str
                        : std::string("unexpected token ") + yellow(token_.spelling));

    return false;
}

bool Parser::expect(tok tok, std::string const& str) {
    if (tok == token_.type) {
        return true;
    }

    error(
        str.size() > 0 ? str
                       : std::string("unexpected token ") + yellow(token_.spelling));

    return false;
}

void Parser::error(std::string msg) {
    std::string location_info = pprintf(
        "%:% ", module_ ? module_->source_name() : "", token_.location);
    if (status_ == lexerStatus::error) {
        // append to current string
        error_string_ += "\n" + white(location_info) + "\n  " + msg;
    }
    else {
        error_string_ = white(location_info) + "\n  " + msg;
        status_ = lexerStatus::error;
    }
}

void Parser::error(std::string msg, Location loc) {
    std::string location_info = pprintf(
        "%:% ", module_ ? module_->source_name() : "", loc);
    if (status_ == lexerStatus::error) {
        // append to current string
        error_string_ += "\n" + green(location_info) + msg;
    }
    else {
        error_string_ = green(location_info) + msg;
        status_ = lexerStatus::error;
    }
}

Parser::Parser(Module& m, bool advance):
    Lexer(m.buffer()),
    module_(&m) {
    // prime the first token
    get_token();

    if (advance) {
        parse();
    }
}

Parser::Parser(std::string const& buf):
    Lexer(buf),
    module_(nullptr) {
    // prime the first token
    get_token();
}

bool Parser::parse() {
    // perform first pass to read the descriptive blocks and
    // record the location of the verb blocks
    while (token_.type != tok::eof) {
        switch (token_.type) {
        case tok::title:
            parse_title();
            break;
        case tok::neuron:
            parse_neuron_block();
            break;
        case tok::state:
            parse_state_block();
            break;
        case tok::units:
            parse_units_block();
            break;
        case tok::constant:
            parse_constant_block();
            break;
        case tok::parameter:
            parse_parameter_block();
            break;
        case tok::assigned:
            parse_assigned_block();
            break;
        case tok::white_noise:
            parse_white_noise_block();
            break;
        // INITIAL, KINETIC, DERIVATIVE, PROCEDURE, NET_RECEIVE and BREAKPOINT blocks
        // are all lowered to ProcedureExpression
        case tok::net_receive:
        case tok::breakpoint:
        case tok::initial:
        case tok::post_event:
        case tok::kinetic:
        case tok::linear:
        case tok::derivative:
        case tok::procedure: {
            auto p = parse_procedure();
            if (!p) break;
            module_->add_callable(std::move(p));
        } break;
        case tok::function: {
            auto f = parse_function();
            if (!f) break;
            module_->add_callable(std::move(f));
        } break;
        default:
            error(pprintf("expected block type, found '%'", token_.spelling));
            break;
        }
        if (status() == lexerStatus::error) {
            std::cerr << red("error: ") << error_string_ << std::endl;
            return false;
        }
    }

    return true;
}

// consume a comma separated list of identifiers
// NOTE: leaves the current location at begining of the last identifier in the list
// OK:  empty list ""
// OK:  list with one identifier "a"
// OK:  list with mutiple identifier "a, b, c, d"
// BAD: list with keyword "a, b, else, d"
// list with trailing comma "a, b,\n"
// list with keyword "a, if, b"
std::vector<Token> Parser::comma_separated_identifiers() {
    std::vector<Token> tokens;
    int startline = location_.line;
    // handle is an empty list at the end of a line
    if (peek().location.line > startline) {
        // this happens when scanning WRITE below:
        //      USEION k READ a, b WRITE
        // leave to the caller to decide whether an empty list is an error
        return tokens;
    }
    while (1) {
        get_token();

        // first check if a new line was encounterd
        if (location_.line > startline) {
            return tokens;
        }
        else if (token_.type == tok::identifier) {
            tokens.push_back(token_);
        }
        else if (is_keyword(token_)) {
            error(pprintf("found keyword '%', expected a variable name", token_.spelling));
            return tokens;
        }
        else if (token_.type == tok::real || token_.type == tok::integer) {
            error(pprintf("found number '%', expected a variable name", token_.spelling));
            return tokens;
        }
        else {
            error(pprintf("found '%', expected a variable name", token_.spelling));
            return tokens;
        }

        // look ahead to check for a comma.  This approach ensures that the
        // first token after the end of the list is not consumed
        if (peek().type == tok::comma) {
            // load the comma
            get_token();
            // assert that the list can't run off the end of a line
            if (peek().location.line > startline) {
                error("line can't end with a '" + yellow(",") + "'");
                return tokens;
            }
        }
        else {
            break;
        }
    }
    get_token(); // prime the first token after the list

    return tokens;
}

/*
NEURON {
   THREADSAFE
   SUFFIX KdShu2007
   USEION k WRITE ik READ xy
   RANGE  gkbar, ik, ek
   GLOBAL minf, mtau, hinf, htau
}
*/
void Parser::parse_neuron_block() {
    NeuronBlock neuron_block;

    get_token();

    // assert that the block starts with a curly brace
    if (token_.type != tok::lbrace) {
        error(pprintf("NEURON block must start with a curly brace {, found '%'",
            token_.spelling));
        return;
    }

    // initialize neuron block
    neuron_block.threadsafe = false;

    // there are no use cases for curly brace in a NEURON block, so we don't
    // have to count them we have to get the next token before entering the loop
    // to handle the case of an empty block {}
    get_token();
    while (token_.type != tok::rbrace) {
        switch (token_.type) {
        case tok::threadsafe:
            neuron_block.threadsafe = true;
            get_token(); // consume THREADSAFE
            break;

        case tok::suffix:
        case tok::point_process:
        case tok::voltage_process:
        case tok::junction_process:
            neuron_block.kind = (token_.type == tok::suffix) ? moduleKind::density :
                                (token_.type == tok::voltage_process) ? moduleKind::voltage :
                                (token_.type == tok::point_process) ? moduleKind::point : moduleKind::junction;

            // set the module kind
            module_->kind(neuron_block.kind);

            get_token(); // consume SUFFIX / POINT_PROCESS
            // assert that a valid name for the Neuron has been specified
            if (token_.type != tok::identifier) {
                error(pprintf("invalid name for mechanism, found '%'", token_.spelling));
                return;
            }
            neuron_block.name = token_.spelling;

            get_token(); // consume the name
            break;

        // this will be a comma-separated list of identifiers
        case tok::global:
            // the ranges are a comma-seperated list of identifiers
            {
                auto identifiers = comma_separated_identifiers();
                // bail if there was an error reading the list
                if (status_ == lexerStatus::error) {
                    return;
                }
                for (auto const& id: identifiers) {
                    neuron_block.globals.push_back(id);
                }
            }
            break;

        // this will be a comma-separated list of identifiers
        case tok::range:
            // the ranges are a comma-seperated list of identifiers
            {
                auto identifiers = comma_separated_identifiers();
                if (status_ == lexerStatus::error) { // bail if there was an error reading the list
                    return;
                }
                for (auto const& id: identifiers) {
                    neuron_block.ranges.push_back(id);
                }
            }
            break;

        case tok::useion: {
            IonDep ion;
            // we have to parse the name of the ion first
            get_token();
            // check this is an identifier token
            if (token_.type != tok::identifier) {
                error(pprintf("invalid name for an ion chanel '%'", token_.spelling));
                return;
            }

            ion.name = token_.spelling;
            get_token(); // consume the ion name

            // this loop ensures that we don't gobble any tokens past
            // the end of the USEION clause
            while (token_.type == tok::read || token_.type == tok::write) {
                auto& target = (token_.type == tok::read) ? ion.read
                                                          : ion.write;
                std::vector<Token> identifiers = comma_separated_identifiers();
                // bail if there was an error reading the list
                if (status_ == lexerStatus::error) {
                    return;
                }
                for (auto const& id: identifiers) {
                    target.push_back(id);
                }
            }

            if (token_.type == tok::valence) {
                ion.has_valence_expr = true;

                // consume "Valence"
                get_token();

                // take and consume variable name or signed integer
                if (token_.type == tok::identifier) {
                    ion.valence_var = token_;
                    get_token();
                }
                else {
                    ion.expected_valence = value_signed_integer();
                }
            }

            // add the ion dependency to the NEURON block
            neuron_block.ions.push_back(std::move(ion));
        } break;

        case tok::nonspecific_current:
            // Assume that there is one non-specific current per mechanism.
            // It would be easy to extend this to multiple currents,
            // however there are no mechanisms in the CoreNeuron repository
            // that do this
            {
                get_token(); // consume NONSPECIFIC_CURRENT

                auto tok = token_;

                // parse the current name and check for errors
                auto id = parse_identifier();
                if (status_ == lexerStatus::error) {
                    return;
                }

                // store the token with nonspecific current's name and location
                neuron_block.nonspecific_current = tok;
            }
            break;

        // the parser encountered an invalid symbol
        default:
            error(pprintf("there was an invalid statement '%' in NEURON block",
                token_.spelling));
            return;
        }
    }

    // copy neuron block into module
    module_->neuron_block(neuron_block);

    // now we have a curly brace, so prime the next token
    get_token();
}

void Parser::parse_state_block() {
    StateBlock state_block;

    get_token();

    // assert that the block starts with a curly brace
    if (token_.type != tok::lbrace) {
        error(pprintf("STATE block must start with a curly brace {, found '%'", token_.spelling));
        return;
    }

    // there are no use cases for curly brace in a STATE block, so we don't have
    // to count them we have to get the next token before entering the loop to
    // handle the case of an empty block {}
    get_token();
    while (token_.type != tok::rbrace && token_.type != tok::eof) {
        int line = location_.line;
        Id parm;

        if (token_.type != tok::identifier) {
            error(pprintf("'%' is not a valid name for a state variable",
                token_.spelling));
            return;
        }

        parm.token = token_;
        get_token();

        if (token_.type == tok::from) {
            // silently skips from/to
            from_to_description();
            if (status_ == lexerStatus::error) {
                return;
            }
        }

        // get unit parameters
        if (line == location_.line && token_.type == tok::lparen) {
            parm.units = unit_description();
            if (status_ == lexerStatus::error) {
                error(pprintf("STATUS block unexpected symbol '%s'",
                    token_.spelling));
                return;
            }
        }

        state_block.state_variables.push_back(parm);
    }

    // add this state block information to the module
    module_->state_block(state_block);

    // now we have a curly brace, so prime the next token
    get_token();
}

// scan a unit block
void Parser::parse_units_block() {
    UnitsBlock units_block;

    get_token();

    // assert that the block starts with a curly brace
    if (token_.type != tok::lbrace) {
        error(pprintf("UNITS block must start with a curly brace {, found '%'", token_.spelling));
        return;
    }

    // there are no use cases for curly brace in a UNITS block, so we don't have to count them
    get_token();
    while (token_.type != tok::rbrace) {
        // get the alias
        std::vector<Token> lhs = unit_description();
        if (status_ != lexerStatus::happy) return;

        // consume the '=' sign
        if (token_.type != tok::eq) {
            error(pprintf("expected '=', found '%'", token_.spelling));
            return;
        }

        get_token(); // next token

        // get the units
        std::vector<Token> rhs = unit_description();
        if (status_ != lexerStatus::happy) return;

        // store the unit definition
        units_block.unit_aliases.push_back({lhs, rhs});
    }

    // add this state block information to the module
    module_->units_block(units_block);

    // now we have a curly brace, so prime the next token
    get_token();
}

//////////////////////////////////////////////////////
// the parameter block describes variables that are
// to be used as parameters. Some are given values,
// others are simply listed, and some have units
// assigned to them. Here we want to get a list of the
// parameter names, along with values if given.
// We also store the token that describes the units
//////////////////////////////////////////////////////
void Parser::parse_parameter_block() {
    ParameterBlock block;

    get_token();

    // assert that the block starts with a curly brace
    if (token_.type != tok::lbrace) {
        error(pprintf("PARAMETER block must start with a curly brace {, found '%'", token_.spelling));
        return;
    }

    int success = 1;
    // there are no use cases for curly brace in a UNITS block, so we don't have to count them
    get_token();
    while (token_.type != tok::rbrace && token_.type != tok::eof) {
        int line = location_.line;
        Id parm;

        // read the parameter name
        if (token_.type != tok::identifier) {
            success = 0;
            goto parm_exit;
        }
        parm.token = token_; // save full token

        get_token();

        // look for equality
        if (token_.type == tok::eq) {
            get_token(); // consume '='
            parm.value = value_literal();
            if (status_ == lexerStatus::error) {
                success = 0;
                goto parm_exit;
            }
        }

        // get the units
        if (line == location_.line && token_.type == tok::lparen) {
            parm.units = unit_description();
            if (status_ == lexerStatus::error) {
                success = 0;
                goto parm_exit;
            }
        }

        // get the range
        if (line == location_.line && token_.type == tok::lt) {
            parm.range = range_description();
            if (status_ == lexerStatus::error) {
                success = 0;
                goto parm_exit;
            }
        }
        block.parameters.push_back(parm);
    }

    // error if EOF before closing curly brace
    if (token_.type == tok::eof) {
        error("PARAMETER block must have closing '}'");
        goto parm_exit;
    }

    get_token(); // consume closing brace

    module_->parameter_block(block);

parm_exit:
    // only write error message if one hasn't already been logged by the lexer
    if (!success && status_ == lexerStatus::happy) {
        error(pprintf("PARAMETER block unexpected symbol '%s'", token_.spelling));
    }
    return;
}

void Parser::parse_constant_block() {
    get_token();

    // assert that the block starts with a curly brace
    if (token_.type != tok::lbrace) {
        error(pprintf("CONSTANT block must start with a curly brace {, found '%'", token_.spelling));
        return;
    }

    get_token();
    while (token_.type != tok::rbrace && token_.type != tok::eof) {
        int line = location_.line;
        std::string name, value;

        // read the constant name
        if (token_.type != tok::identifier) {
            error(pprintf("CONSTANT block unexpected symbol '%s'", token_.spelling));
            return;
        }
        name = token_.spelling; // save full token

        get_token();

        // look for equality
        if (token_.type == tok::eq) {
            get_token(); // consume '='
            value = value_literal();
            if (status_ == lexerStatus::error) {
                return;
            }
        }

        // get the units
        if (line == location_.line && token_.type == tok::lparen) {
            unit_description();
            if (status_ == lexerStatus::error) {
                return;
            }
        }

        constants_map_.insert({name, value});
    }

    // error if EOF before closing curly brace
    if (token_.type == tok::eof) {
        error("CONSTANT block must have closing '}'");
        return;
    }

    get_token(); // consume closing brace

    return;
}

void Parser::parse_assigned_block() {
    AssignedBlock block;

    get_token();

    // assert that the block starts with a curly brace
    if (token_.type != tok::lbrace) {
        error(pprintf("ASSIGNED block must start with a curly brace {, found '%'", token_.spelling));
        return;
    }

    int success = 1;

    // there are no use cases for curly brace in an ASSIGNED block, so we don't have to count them
    get_token();
    while (token_.type != tok::rbrace && token_.type != tok::eof) {
        int line = location_.line;
        std::vector<Token> variables; // we can have more than one variable on a line

        // the first token must be ...
        if (token_.type != tok::identifier) {
            success = 0;
            goto ass_exit;
        }
        // read all of the identifiers until we run out of identifiers or reach a new line
        while (token_.type == tok::identifier && line == location_.line) {
            variables.push_back(token_);
            get_token();
        }

        // there are some parameters at the end of the line
        if (line == location_.line && token_.type == tok::lparen) {
            auto u = unit_description();
            if (status_ == lexerStatus::error) {
                success = 0;
                goto ass_exit;
            }
            for (auto const& t: variables) {
                block.parameters.push_back(Id(t, "", u));
            }
        }
        else {
            for (auto const& t: variables) {
                block.parameters.push_back(Id(t, "", {}));
            }
        }
    }

    // error if EOF before closing curly brace
    if (token_.type == tok::eof) {
        error("ASSIGNED block must have closing '}'");
        goto ass_exit;
    }

    get_token(); // consume closing brace

    module_->assigned_block(block);

ass_exit:
    // only write error message if one hasn't already been logged by the lexer
    if (!success && status_ == lexerStatus::happy) {
        error(pprintf("ASSIGNED block unexpected symbol '%'", token_.spelling));
    }
    return;
}

void Parser::parse_white_noise_block() {
    WhiteNoiseBlock block;

    get_token();

    // assert that the block starts with a curly brace
    if (token_.type != tok::lbrace) {
        error(pprintf("WHITE_NOISE block must start with a curly brace {, found '%'", token_.spelling));
        return;
    }

    int success = 1;

    // there are no use cases for curly brace in an WHITE_NOISE block, so we don't have to count them
    get_token();
    while (token_.type != tok::rbrace && token_.type != tok::eof) {
        int line = location_.line;
        std::vector<Token> variables; // we can have more than one variable on a line

        // the first token must be ...
        if (token_.type != tok::identifier) {
            success = 0;
            goto wn_exit;
        }
        // read all of the identifiers until we run out of identifiers or reach a new line
        while (token_.type == tok::identifier && line == location_.line) {
            variables.push_back(token_);
            get_token();
        }

        // there must be no paramters at the end of the line
        if (line == location_.line && token_.type == tok::lparen) {
            success = 0;
            goto wn_exit;
        }
        else {
            for (auto const& t: variables) {
                block.parameters.push_back(Id(t, "", {}));
            }
        }
    }

    // error if EOF before closing curly brace
    if (token_.type == tok::eof) {
        error("WHITE_NOISE block must have closing '}'");
        goto wn_exit;
    }

    get_token(); // consume closing brace

    module_->white_noise_block(block);

wn_exit:
    // only write error message if one hasn't already been logged by the lexer
    if (!success && status_ == lexerStatus::happy) {
        error(pprintf("WHITE_NOISE block unexpected symbol '%'", token_.spelling));
    }
    return;
}

// Parse a value (integral or real) with possible preceding unary minus,
// and return as a string.
std::string Parser::value_literal() {
    bool negate = false;

    if (token_.type == tok::minus) {
        negate = true;
        get_token();
    }

    if (constants_map_.find(token_.spelling) != constants_map_.end()) {
        // Remove double negation
        auto v = constants_map_.at(token_.spelling);
        if (v.at(0) == '-' && negate) {
            v.erase(0, 1);
            negate = false;
        }
        auto value = negate ? "-" + v : v;
        get_token();
        return value;
    }

    if (token_.type != tok::integer && token_.type != tok::real) {
        error(pprintf("numeric constant not an integer or real number '%'", token_));
        return "";
    }
    else {
        auto value = negate ? "-" + token_.spelling : token_.spelling;
        get_token();
        return value;
    }
}

// Parse an integral value with possible preceding unary plus or minus,
// and return as an int.
int Parser::value_signed_integer() {
    std::string value;

    if (token_.type == tok::minus) {
        value = "-";
        get_token();
    }
    else if (token_.type == tok::plus) {
        get_token();
    }
    if (token_.type != tok::integer) {
        error(pprintf("numeric constant not an integer '%'", token_));
        return 0;
    }
    else {
        value += token_.spelling;
        get_token();
        return std::stoi(value);
    }
}

std::vector<Token> Parser::unit_description() {
    static const tok legal_tokens[] = {tok::identifier, tok::divide, tok::real, tok::integer};
    int startline = location_.line;
    std::vector<Token> tokens;

    // check that we start with a left parenthesis
    if (token_.type != tok::lparen) {
        error(pprintf("unit description must start with a parenthesis '%'", token_));
        goto unit_exit;
    }

    get_token();

    while (token_.type != tok::rparen) {
        // check for illegal tokens or a new line
        if (!is_in(token_.type, legal_tokens) || startline < location_.line) {
            error(pprintf("incorrect unit description '%'", token_));
            goto unit_exit;
        }

        // add this token to the set
        tokens.push_back(token_);
        get_token();
    }
    // remove trailing right parenthesis ')'
    get_token();

unit_exit:
    return tokens;
}

std::pair<std::string, std::string> Parser::range_description() {
    std::string lb, ub;

    if (token_.type != tok::lt) {
        error(pprintf("range description must start with a left angle bracket '%'", token_));
        return {};
    }

    get_token();
    lb = value_literal();

    if (token_.type != tok::comma) {
        error(pprintf("range description must separate lower and upper bound with a comma '%'", token_));
        return {};
    }

    get_token();
    ub = value_literal();

    if (token_.type != tok::gt) {
        error(pprintf("range description must end with a right angle bracket '%'", token_));
        return {};
    }

    get_token();
    return {lb, ub};
}

std::pair<std::string, std::string> Parser::from_to_description() {
    std::string lb, ub;

    if (token_.type != tok::from) {
        error(pprintf("range description must be of form FROM <number> TO <number>, found '%'", token_));
        return {};
    }

    get_token();
    lb = value_literal();

    if (token_.type != tok::to) {
        error(pprintf("range description must be of form FROM <number> TO <number>, found '%'", token_));
        return {};
    }

    get_token();
    ub = value_literal();

    return {lb, ub};
}

// Returns a prototype expression for a function or procedure call
// Takes an optional argument that allows the user to specify the
// name of the prototype, which is used for prototypes where the name
// is implcitly defined (e.g. INITIAL and BREAKPOINT blocks)
expression_ptr Parser::parse_prototype(std::string name = std::string()) {
    Token identifier = token_;

    if (name.size()) {
        // we assume that the current token_ is still pointing at
        // the keyword, i.e. INITIAL or BREAKPOINT
        identifier.type = tok::identifier;
        identifier.spelling = name;
    }

    // load the parenthesis
    get_token();

    // check for an argument list enclosed in parenthesis (...)
    // return a prototype with an empty argument list if not found
    if (token_.type != tok::lparen) {
        return expression_ptr{new PrototypeExpression(identifier.location, identifier.spelling, {})};
    }

    get_token(); // consume '('
    std::vector<Token> arg_tokens;
    while (token_.type != tok::rparen) {
        // check identifier
        if (token_.type != tok::identifier) {
            error("expected a valid identifier, found '" + yellow(token_.spelling) + "'");
            return nullptr;
        }

        arg_tokens.push_back(token_);

        get_token(); // consume the identifier

        // args may have a unit attached
        if (token_.type == tok::lparen) {
            unit_description();
            if (status_ == lexerStatus::error) {
                return {};
            }
        }

        // look for a comma
        if (!(token_.type == tok::comma || token_.type == tok::rparen)) {
            error("expected a comma or closing parenthesis, found '" + yellow(token_.spelling) + "'");
            return nullptr;
        }

        if (token_.type == tok::comma) {
            get_token(); // consume ','
        }
    }

    if (token_.type != tok::rparen) {
        error("procedure argument list must have closing parenthesis ')'");
        return nullptr;
    }
    get_token(); // consume closing parenthesis

    // pack the arguments into LocalDeclarations
    std::vector<expression_ptr> arg_expressions;
    for (auto const& t: arg_tokens) {
        arg_expressions.emplace_back(make_expression<ArgumentExpression>(t.location, t));
    }

    return make_expression<PrototypeExpression>(identifier.location, identifier.spelling, std::move(arg_expressions));
}

void Parser::parse_title() {
    std::string title;
    int this_line = location().line;

    Token tkn = peek();
    while (tkn.location.line == this_line && tkn.type != tok::eof && status_ == lexerStatus::happy) {
        get_token();
        title += token_.spelling;
        tkn = peek();
    }

    // set the module title
    module_->title(title);

    // load next token
    get_token();
}

/// parse a procedure
/// can handle both PROCEDURE and INITIAL blocks
/// an initial block is stored as a procedure with name 'initial' and empty argument list
symbol_ptr Parser::parse_procedure() {
    expression_ptr p;
    procedureKind kind = procedureKind::normal;

    switch (token_.type) {
    case tok::derivative:
        kind = procedureKind::derivative;
        get_token(); // consume keyword token
        if (!expect(tok::identifier)) return nullptr;
        p = parse_prototype();
        break;
    case tok::kinetic:
        kind = procedureKind::kinetic;
        get_token(); // consume keyword token
        if (!expect(tok::identifier)) return nullptr;
        p = parse_prototype();
        break;
    case tok::linear:
        kind = procedureKind::linear;
        get_token(); // consume keyword token
        if (!expect(tok::identifier)) return nullptr;
        p = parse_prototype();
        break;
    case tok::procedure:
        kind = procedureKind::normal;
        get_token(); // consume keyword token
        if (!expect(tok::identifier)) return nullptr;
        p = parse_prototype();
        break;
    case tok::initial:
        kind = procedureKind::initial;
        p = parse_prototype("initial");
        break;
    case tok::breakpoint:
        kind = procedureKind::breakpoint;
        p = parse_prototype("breakpoint");
        break;
    case tok::net_receive:
        kind = procedureKind::net_receive;
        p = parse_prototype("net_receive");
        break;
    case tok::post_event:
        kind = procedureKind::post_event;
        p = parse_prototype("post_event");
        break;
    default:
        // it is a compiler error if trying to parse_procedure() without
        // having DERIVATIVE, KINETIC, PROCEDURE, INITIAL or BREAKPOINT keyword
        throw compiler_exception(
            "attempt to parse_procedure() without {DERIVATIVE,KINETIC,PROCEDURE,INITIAL,BREAKPOINT}",
            location_);
    }
    if (p == nullptr) return nullptr;

    // check for opening left brace {
    if (!expect(tok::lbrace)) return nullptr;

    // parse the body of the function
    expression_ptr body = parse_block(false);
    if (body == nullptr) return nullptr;

    auto proto = p->is_prototype();
    if(kind == procedureKind::net_receive) {
        return make_symbol<NetReceiveExpression> (proto->location(), proto->name(), std::move(proto->args()), std::move(body));
    }
    if(kind == procedureKind::post_event) {
        return make_symbol<PostEventExpression> (proto->location(), proto->name(), std::move(proto->args()), std::move(body));
    }
    return make_symbol<ProcedureExpression> (proto->location(), proto->name(), std::move(proto->args()), std::move(body), kind);
}

symbol_ptr Parser::parse_function() {
    get_token(); // consume FUNCTION token

    // check that a valid identifier name was specified by the user
    if (!expect(tok::identifier)) return nullptr;

    // parse the prototype
    auto p = parse_prototype();
    if (p == nullptr) return nullptr;

    // Functions may have a unit attached
    if (token_.type == tok::lparen) {
        unit_description();
        if (status_ == lexerStatus::error) {
            return {};
        }
    }

    // check for opening left brace {
    if (!expect(tok::lbrace)) return nullptr;

    // parse the body of the function
    auto body = parse_block(false);
    if (body == nullptr) return nullptr;

    PrototypeExpression* proto = p->is_prototype();
    return make_symbol<FunctionExpression>(proto->location(), proto->name(), std::move(proto->args()), std::move(body));
}

// this is the first port of call when parsing a new line inside a verb block
// it tests to see whether the expression is:
//      :: LOCAL identifier
//      :: expression
expression_ptr Parser::parse_statement() {
    switch (token_.type) {
    case tok::if_stmt:
        return parse_if();
        break;
    case tok::conductance:
        return parse_conductance();
    case tok::solve:
        return parse_solve();
    case tok::local:
        return parse_local();
    case tok::watch:
        return parse_watch();
    case tok::identifier:
        return parse_line_expression();
    case tok::conserve:
        return parse_conserve_expression();
    case tok::compartment:
        return parse_compartment_statement();
    case tok::tilde:
        return parse_tilde_expression();
    case tok::initial:
        // only used for INITIAL block in NET_RECEIVE
        return parse_initial();
    default:
        error(pprintf("unexpected token type % '%'", token_string(token_.type), token_.spelling));
        return nullptr;
    }
    return nullptr;
}

expression_ptr Parser::parse_identifier() {
    if (constants_map_.find(token_.spelling) != constants_map_.end()) {
        // save location and value of the identifier
        auto id = make_expression<NumberExpression>(token_.location, constants_map_.at(token_.spelling));

        // consume the number
        get_token();

        // return the value of the constant
        return id;
    }
    // save name and location of the identifier
    auto id = make_expression<IdentifierExpression>(token_.location, token_.spelling);

    // consume identifier
    get_token();

    // return variable identifier
    return id;
}

expression_ptr Parser::parse_call() {
    // save name and location of the identifier
    Token idtoken = token_;

    // consume identifier
    get_token();

    // check for a function call
    // assert this is so
    if (token_.type != tok::lparen) {
        throw compiler_exception(
            "should not be parsing parse_call without trailing '('",
            location_);
    }

    std::vector<expression_ptr> args;

    // parse a function call
    get_token(); // consume '('

    while (token_.type != tok::rparen) {
        auto e = parse_expression();
        if (!e) return e;

        args.emplace_back(std::move(e));

        // reached the end of the argument list
        if (token_.type == tok::rparen) break;

        // insist on a comma between arguments
        if (!expect(tok::comma, "call arguments must be separated by ','"))
            return expression_ptr();
        get_token(); // consume ','
    }

    // check that we have a closing parenthesis
    if (!expect(tok::rparen, "function call missing closing ')'")) {
        return expression_ptr();
    }
    get_token(); // consume ')'

    return make_expression<CallExpression>(idtoken.location, idtoken.spelling, std::move(args));
}

// parse a full line expression, i.e. one of
//      :: procedure call        e.g. rates(v+0.01)
//      :: assignment expression e.g. x = y + 3
// to parse a subexpression, see parse_expression()
// proceeds by first parsing the LHS (which may be a variable or function call)
// then attempts to parse the RHS if
//      1. the lhs is not a procedure call
//      2. the operator that follows is =
expression_ptr Parser::parse_line_expression() {
    int line = location_.line;
    expression_ptr lhs;
    Token next = peek();
    if (next.type == tok::lparen) {
        lhs = parse_call();
        // we have to ensure that a procedure call is alone on the line
        // to avoid :
        //      :: assigning to it            e.g. foo() = x + 6
        //      :: stray symbols coming after e.g. foo() + x
        // We assume that foo is a procedure call, if it is an eroneous
        // function call this has to be caught in the second pass.
        // or optimized away with a warning
        if (!lhs) return lhs;
        if (location_.line == line && token_.type != tok::eof) {
            error(pprintf(
                "expected a new line after call expression, found '%'",
                yellow(token_.spelling)));
            return expression_ptr();
        }
        return lhs;
    }
    else if (next.type == tok::prime) {
        lhs = make_expression<DerivativeExpression>(location_, token_.spelling);
        // consume both name and derivative operator
        get_token();
        get_token();
        // a derivative statement must be followed by '='
        if (token_.type != tok::eq) {
            error("a derivative declaration must have an assignment of the "
                  "form\n  x' = expression\n  where x is a state variable");
            return expression_ptr();
        }
    }
    else {
        lhs = parse_unaryop();
    }

    if (!lhs) { // error
        return lhs;
    }

    // we parse a binary expression if followed by an operator
    if (token_.type == tok::eq) {
        Token op = token_; // save the '=' operator with location
        get_token();       // consume the '=' operator
        return parse_binop(std::move(lhs), op);
    }
    else if (line == location_.line && token_.type != tok::eof) {
        error(pprintf("expected an assignment '%' or new line, found '%'",
            yellow("="),
            yellow(token_.spelling)));
        return nullptr;
    }

    return lhs;
}

expression_ptr Parser::parse_stoich_term() {
    expression_ptr coeff = make_expression<IntegerExpression>(location_, 1);
    auto here = location_;
    bool negative = false;

    while (token_.type == tok::minus) {
        negative = !negative;
        get_token(); // consume '-'
    }

    if (token_.type == tok::integer) {
        coeff = parse_integer();
    }

    if (token_.type != tok::identifier) {
        error(pprintf("expected an identifier, found '%'", yellow(token_.spelling)));
        return nullptr;
    }

    if (negative) {
        coeff = make_expression<IntegerExpression>(here, -coeff->is_integer()->integer_value());
    }
    return make_expression<StoichTermExpression>(here, std::move(coeff), parse_identifier());
}

expression_ptr Parser::parse_stoich_expression() {
    std::vector<expression_ptr> terms;
    auto here = location_;

    if (token_.type == tok::integer || token_.type == tok::identifier || token_.type == tok::minus) {
        auto term = parse_stoich_term();
        if (!term) return nullptr;

        terms.push_back(std::move(term));

        while (token_.type == tok::plus || token_.type == tok::minus) {
            if (token_.type == tok::plus) {
                get_token(); // consume plus
            }

            auto term = parse_stoich_term();
            if (!term) return nullptr;

            terms.push_back(std::move(term));
        }
    }

    return make_expression<StoichExpression>(here, std::move(terms));
}

expression_ptr Parser::parse_tilde_expression() {
    auto here = location_;

    if (token_.type != tok::tilde) {
        error(pprintf("expected '%', found '%'", yellow("~"), yellow(token_.spelling)));
        return nullptr;
    }
    get_token(); // consume tilde

    if (search_to_eol(tok::arrow)) {
        expression_ptr lhs = parse_stoich_expression();
        if (!lhs) return nullptr;

        // reaction halves must comprise non-negative terms
        for (const auto& term: lhs->is_stoich()->terms()) {
            // should always be true
            if (auto sterm = term->is_stoich_term()) {
                if (sterm->negative()) {
                    error(pprintf("expected only non-negative terms in reaction lhs, found '%'",
                        yellow(term->to_string())));
                    return nullptr;
                }
            }
        }

        if (token_.type != tok::arrow) {
            error(pprintf("expected '%', found '%'", yellow("<->"), yellow(token_.spelling)));
            return nullptr;
        }

        get_token(); // consume arrow
        expression_ptr rhs = parse_stoich_expression();
        if (!rhs) return nullptr;

        for (const auto& term: rhs->is_stoich()->terms()) {
            // should always be true
            if (auto sterm = term->is_stoich_term()) {
                if (sterm->negative()) {
                    error(pprintf("expected only non-negative terms in reaction rhs, found '%'",
                        yellow(term->to_string())));
                    return nullptr;
                }
            }
        }

        if (token_.type != tok::lparen) {
            error(pprintf("expected '%', found '%'", yellow("("), yellow(token_.spelling)));
            return nullptr;
        }

        get_token(); // consume lparen
        expression_ptr fwd = parse_expression();
        if (!fwd) return nullptr;

        if (token_.type != tok::comma) {
            error(pprintf("expected '%', found '%'", yellow(","), yellow(token_.spelling)));
            return nullptr;
        }

        get_token(); // consume comma
        expression_ptr rev = parse_expression();
        if (!rev) return nullptr;

        if (token_.type != tok::rparen) {
            error(pprintf("expected '%', found '%'", yellow(")"), yellow(token_.spelling)));
            return nullptr;
        }

        get_token(); // consume rparen
        return make_expression<ReactionExpression>(here, std::move(lhs), std::move(rhs), std::move(fwd), std::move(rev));
    }
    else if (search_to_eol(tok::eq)) {
        auto lhs_bin = parse_expression(tok::eq);

        if (token_.type != tok::eq) {
            error(pprintf("expected '%', found '%'", yellow("="), yellow(token_.spelling)));
            return nullptr;
        }

        get_token(); // consume =
        auto rhs = parse_expression();
        return make_expression<LinearExpression>(here, std::move(lhs_bin), std::move(rhs));
    }
    else {
        error(pprintf("expected stoichiometric or linear expression, found neither"));
        return nullptr;
    }
}

expression_ptr Parser::parse_conserve_expression() {
    auto here = location_;

    if (token_.type != tok::conserve) {
        error(pprintf("expected '%', found '%'", yellow("CONSERVE"), yellow(token_.spelling)));
        return nullptr;
    }

    get_token(); // consume 'CONSERVE'
    auto lhs = parse_stoich_expression();
    if (!lhs) return nullptr;

    if (token_.type != tok::eq) {
        error(pprintf("expected '%', found '%'", yellow("="), yellow(token_.spelling)));
        return nullptr;
    }

    get_token(); // consume '='
    auto rhs = parse_expression();
    if (!rhs) return nullptr;

    return make_expression<ConserveExpression>(here, std::move(lhs), std::move(rhs));
}

expression_ptr Parser::parse_expression(int prec, tok stop_token) {
    auto lhs = parse_unaryop();
    if (lhs == nullptr) return nullptr;

    // Combine all sub-expressions with precedence greater than prec.
    for (;;) {
        if (token_.type == stop_token) {
            return lhs;
        }

        auto op = token_;
        auto p_op = binop_precedence(op.type);

        // Note: all tokens that are not infix binary operators have
        // precedence of -1, so expressions like function calls will short
        // circuit this loop here.
        if (p_op <= prec) return lhs;

        get_token(); // consume the infix binary operator

        lhs = parse_binop(std::move(lhs), op);
        if (!lhs) return nullptr;
    }

    return lhs;
}

expression_ptr Parser::parse_expression() {
    return parse_expression(0);
}

expression_ptr Parser::parse_expression(tok t) {
    return parse_expression(0, t);
}

/// Parse a unary expression.
/// If called when the current node in the AST is not a unary expression the call
/// will be forwarded to parse_primary. This mechanism makes it possible to parse
/// all nodes in the expression using parse_unary, which simplifies the call sites
/// with either a primary or unary node is to be parsed.
/// It also simplifies parsing nested unary functions, e.g. x + - - y
expression_ptr Parser::parse_unaryop() {
    expression_ptr e;
    Token op = token_;
    switch (token_.type) {
    case tok::plus:
        // plus sign is simply ignored
        get_token(); // consume '+'
        return parse_unaryop();
    case tok::minus:
        get_token();         // consume '-'
        e = parse_unaryop(); // handle recursive unary
        if (!e) return nullptr;
        return unary_expression(token_.location, op.type, std::move(e));
    case tok::exp:
    case tok::sin:
    case tok::cos:
    case tok::log:
    case tok::abs:
    case tok::safeinv:
    case tok::exprelr:
    case tok::sqrt:
    case tok::step_right:
    case tok::step_left:
    case tok::step:
    case tok::signum:
        get_token(); // consume operator (exp, sin, cos or log)
        if (token_.type != tok::lparen) {
            error("missing parenthesis after call to " + yellow(op.spelling));
            return nullptr;
        }
        e = parse_unaryop(); // handle recursive unary
        if (!e) return nullptr;
        return unary_expression(token_.location, op.type, std::move(e));
    default:
        return parse_primary();
    }
    return nullptr;
}

/// parse a primary expression node
/// expects one of :
///  ::  number
///  ::  identifier
///  ::  watch
///  ::  call
///  ::  parenthesis expression (parsed recursively)
///  ::  prefix binary operators
expression_ptr Parser::parse_primary() {
    switch (token_.type) {
    case tok::real:
        return parse_real();
    case tok::integer:
        return parse_integer();
    case tok::identifier:
        if (peek().type == tok::lparen) {
            return parse_call();
        }
        return parse_identifier();
    case tok::lparen:
        return parse_parenthesis_expression();
    case tok::min:
    case tok::max: {
        auto op = token_;
        // handle infix binary operators, e.g. min(l,r) and max(l,r)
        get_token(); // consume operator keyword token
        if (token_.type != tok::lparen) {
            error("expected opening parenthesis '('");
            return nullptr;
        }
        get_token(); // consume (
        auto lhs = parse_expression();
        if (!lhs) return nullptr;

        if (token_.type != tok::comma) {
            error("expected comma ','");
            return nullptr;
        }
        get_token(); // consume ,

        auto rhs = parse_expression();
        if (!rhs) return nullptr;
        if (token_.type != tok::rparen) {
            error("expected closing parenthesis ')'");
            return nullptr;
        }
        get_token(); // consume )
        return binary_expression(op.location, op.type, std::move(lhs), std::move(rhs));
    }
    default: // fall through to return nullptr at end of function
        error(pprintf("unexpected token '%' in expression",
            yellow(token_.spelling)));
    }

    return nullptr;
}

expression_ptr Parser::parse_parenthesis_expression() {
    // never call unless at start of parenthesis

    if (token_.type != tok::lparen) {
        throw compiler_exception(
            "attempt to parse a parenthesis_expression() without opening parenthesis",
            location_);
    }

    get_token(); // consume '('

    auto e = parse_expression();

    // check for closing parenthesis ')'
    if (!e || !expect(tok::rparen)) return nullptr;

    get_token(); // consume ')'

    return e;
}

expression_ptr Parser::parse_real() {
    auto e = make_expression<NumberExpression>(token_.location, token_.spelling);
    get_token(); // consume the number
    return e;
}

expression_ptr Parser::parse_integer() {
    auto e = make_expression<IntegerExpression>(token_.location, token_.spelling);
    get_token(); // consume the number
    return e;
}

expression_ptr Parser::parse_binop(expression_ptr&& lhs, Token op_left) {
    auto p_op_left = binop_precedence(op_left.type);
    auto rhs = parse_expression(p_op_left);
    if (!rhs) return nullptr;

    auto op_right = token_;
    auto p_op_right = binop_precedence(op_right.type);
    bool right_assoc = operator_associativity(op_right.type) == associativityKind::right;

    if (p_op_right > p_op_left) {
        throw compiler_exception(
            "parse_binop() : encountered operator of higher precedence",
            location_);
    }

    if (p_op_right < p_op_left) {
        return binary_expression(op_left.location, op_left.type, std::move(lhs), std::move(rhs));
    }

    get_token(); // consume op_right
    if (right_assoc) {
        rhs = parse_binop(std::move(rhs), op_right);
        if (!rhs) return nullptr;

        return binary_expression(op_left.location, op_left.type, std::move(lhs), std::move(rhs));
    }
    else {
        lhs = binary_expression(op_left.location, op_left.type, std::move(lhs), std::move(rhs));
        return parse_binop(std::move(lhs), op_right);
    }
}

/// parse a local variable definition
/// a local variable definition is a line with the form
///     LOCAL x
/// where x is a valid identifier name
expression_ptr Parser::parse_local() {
    Location loc = location_;

    get_token(); // consume LOCAL

    // create local expression stub
    auto e = make_expression<LocalDeclaration>(loc);
    if (!e) return e;

    // add symbols
    while (1) {
        if (!expect(tok::identifier)) return nullptr;

        // try adding variable name to list
        if (!e->is_local_declaration()->add_variable(token_)) {
            error(e->error_message());
            return nullptr;
        }
        get_token(); // consume identifier

        // look for comma that indicates continuation of the variable list
        if (token_.type == tok::comma) {
            get_token();
        }
        else {
            break;
        }
    }

    return e;
}

/// parse a SOLVE statement
/// a SOLVE statement specifies a procedure and a method
///     SOLVE procedure METHOD method
/// we also support SOLVE statements without a METHOD clause
/// for backward compatability with performance hacks that
/// are implemented in some key mod files (i.e. Prob* synapses)
expression_ptr Parser::parse_solve() {
    int line = location_.line;
    Location loc = location_; // solve location for expression
    std::string name;
    solverMethod method;
    solverVariant variant;

    get_token(); // consume the SOLVE keyword

    if (token_.type != tok::identifier) goto solve_statement_error;

    name = token_.spelling; // save name of procedure
    get_token();            // consume the procedure identifier

    variant = solverVariant::regular;
    if (token_.type != tok::method && token_.type != tok::steadystate) {
        method = solverMethod::none;
    }
    else {
        if (token_.type == tok::steadystate) {
            variant = solverVariant::steadystate;
        }
        get_token(); // consume the METHOD keyword
        switch (token_.type) {
        case tok::cnexp:
            method = solverMethod::cnexp;
            break;
        case tok::sparse:
            method = solverMethod::sparse;
            break;
        case tok::stochastic:
            method = solverMethod::stochastic;
            break;
        default:
            goto solve_statement_error;
        }

        get_token(); // consume the method description
    }
    // check that the rest of the line was empty
    if (line == location_.line) {
        if (token_.type != tok::eof) goto solve_statement_error;
    }

    return make_expression<SolveExpression>(loc, name, method, variant);

solve_statement_error:
    error("SOLVE statements must have the form\n"
          "  SOLVE x METHOD method\n"
          "    or\n"
          "  SOLVE x STEADYSTATE sparse\n"
          "    or\n"
          "  SOLVE x\n"
          "where 'x' is the name of a DERIVATIVE block and "
          "'method' is 'cnexp' or 'sparse'",
        loc);
    return nullptr;
}

/// parse a CONDUCTANCE statement
/// a CONDUCTANCE statement specifies a variable and a channel
/// where the channel is optional
///     CONDUCTANCE name USEION channel
///     CONDUCTANCE name
expression_ptr Parser::parse_conductance() {
    int line = location_.line;
    Location loc = location_; // solve location for expression
    std::string name;
    std::string channel;

    get_token(); // consume the CONDUCTANCE keyword

    if (token_.type != tok::identifier) goto conductance_statement_error;

    name = token_.spelling; // save name of variable
    get_token();            // consume the variable identifier

    if (token_.type == tok::useion) {
        get_token(); // consume the USEION keyword
        if (token_.type != tok::identifier) goto conductance_statement_error;

        channel = token_.spelling;
        get_token(); // consume the ion channel type
    }
    // check that the rest of the line was empty
    if (line == location_.line) {
        if (token_.type != tok::eof) goto conductance_statement_error;
    }

    return make_expression<ConductanceExpression>(loc, name, channel);

conductance_statement_error:
    error("CONDUCTANCE statements must have the form\n"
          "  CONDUCTANCE g USEION channel\n"
          "    or\n"
          "  CONDUCTANCE g\n"
          "where 'g' is the name of a variable, and 'channel' is the type of ion channel",
        loc);
    return nullptr;
}

// WATCH (cond) flag
expression_ptr Parser::parse_watch() {
    Location loc = location_; // solve location for expression
    get_token();              // consume keyword

    parse_parenthesis_expression();
    parse_expression();

    error("WATCH statements are not supported in modcc.", loc);
    return nullptr;
}


expression_ptr Parser::parse_if() {
    Token if_token = token_;
    get_token(); // consume 'if'

    if (!expect(tok::lparen)) return nullptr;

    // parse the conditional
    auto cond = parse_parenthesis_expression();
    if (!cond) return nullptr;

    // parse the block of the true branch
    auto true_branch = parse_block(true);
    if (!true_branch) return nullptr;

    // parse the false branch if there is an else
    expression_ptr false_branch;
    if (token_.type == tok::else_stmt) {
        get_token(); // consume else

        // handle 'else if {}' case recursively
        if (token_.type == tok::if_stmt) {
            expr_list_type if_block;
            auto exp = parse_if();
            if_block.push_back(std::move(exp));
            false_branch = make_expression<BlockExpression>(Location(), std::move(if_block), true);
        }
        // we have a closing 'else {}'
        else if (token_.type == tok::lbrace) {
            false_branch = parse_block(true);
        }
        else {
            error("expect either '" + yellow("if") + "' or '" + yellow("{") + " after else");
            return nullptr;
        }
    }

    return make_expression<IfExpression>(if_token.location, std::move(cond), std::move(true_branch), std::move(false_branch));
}

// takes a flag indicating whether the block is at procedure/function body,
// or lower. Can be used to check for illegal statements inside a nested block,
// e.g. LOCAL declarations.
expression_ptr Parser::parse_block(bool is_nested) {
    // blocks have to be enclosed in curly braces {}
    expect(tok::lbrace);

    get_token(); // consume '{'

    // save the location of the first statement as the starting point for the block
    Location block_location = token_.location;

    expr_list_type body;
    while (token_.type != tok::rbrace) {
        auto e = parse_statement();
        if (!e) return e;

        if (is_nested) {
            if (e->is_local_declaration()) {
                error("LOCAL variable declarations are not allowed inside a nested scope");
                return nullptr;
            }
            if (e->is_reaction()) {
                error("reaction expressions are not allowed inside a nested scope");
                return nullptr;
            }
        }

        body.emplace_back(std::move(e));
    }

    if (token_.type != tok::rbrace) {
        error(pprintf("could not find closing '%' for else statement that started at ",
            yellow("}"),
            block_location));
        return nullptr;
    }
    get_token(); // consume closing '}'

    return make_expression<BlockExpression>(block_location, std::move(body), is_nested);
}

expression_ptr Parser::parse_initial() {
    // has to start with INITIAL: error in compiler implementaion otherwise
    expect(tok::initial);

    // save the location of the first statement as the starting point for the block
    Location block_location = token_.location;

    get_token(); // consume 'INITIAL'

    if (!expect(tok::lbrace)) return nullptr;
    get_token(); // consume '{'

    expr_list_type body;
    while (token_.type != tok::rbrace) {
        auto e = parse_statement();
        if (!e) return e;

        // disallow variable declarations in an INITIAL block
        if (e->is_local_declaration()) {
            error("LOCAL variable declarations are not allowed inside a nested scope");
            return nullptr;
        }

        body.emplace_back(std::move(e));
    }

    if (token_.type != tok::rbrace) {
        error(pprintf("could not find closing '%' for else statement that started at ",
            yellow("}"),
            block_location));
        return nullptr;
    }
    get_token(); // consume closing '}'

    return make_expression<InitialBlock>(block_location, std::move(body));
}

expression_ptr Parser::parse_compartment_statement() {
    auto here = location_;

    if (token_.type != tok::compartment) {
        error(pprintf("expected '%', found '%'", yellow("COMPARTMENT"), yellow(token_.spelling)));
        return nullptr;
    }

    get_token(); // consume 'COMPARTMENT'
    auto scale_factor = parse_expression(tok::rbrace);
    if (!scale_factor) return nullptr;

    if (token_.type != tok::lbrace) {
        error(pprintf("expected '%', found '%'", yellow("{"), yellow(token_.spelling)));
        return nullptr;
    }

    get_token(); // consume '{'
    std::vector<expression_ptr> states;
    while (token_.type != tok::rbrace) {
        // check identifier
        if (token_.type != tok::identifier) {
            error("expected a valid identifier, found '" + yellow(token_.spelling) + "'");
            return nullptr;
        }

        auto e = make_expression<IdentifierExpression>(token_.location, token_.spelling);
        states.emplace_back(std::move(e));

        get_token(); // consume the identifier
    }
    get_token(); // consume the rbrace
    return make_expression<CompartmentExpression>(here, std::move(scale_factor), std::move(states));
}
