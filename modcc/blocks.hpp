#pragma once

#include <algorithm>
#include <iosfwd>
#include <string>
#include <vector>
#include <set>

#include "identifier.hpp"
#include "location.hpp"
#include "token.hpp"
#include <libmodcc/export.hpp>

// describes a relationship with an ion channel
struct IonDep {
    std::string name;         // name of ion channel
    std::vector<Token> read;  // name of channels parameters to write
    std::vector<Token> write; // name of channels parameters to read

    Token valence_var;        // optional variable name following VALENCE
    int expected_valence = 0; // optional integer following VALENCE
    bool has_valence_expr = false;

    bool has_variable(std::string const& name) const {
        return writes_variable(name) || reads_variable(name);
    };
    bool uses_current() const {
        return has_variable("i" + name);
    };
    bool uses_rev_potential() const {
        return has_variable("e" + name);
    };
    bool uses_concentration_int() const {
        return has_variable(name + "i");
    };
    bool uses_concentration_diff() const {
        return has_variable(name + "d");
    };
    bool uses_concentration_ext() const {
        return has_variable(name + "o");
    };
    bool writes_current() const {
        return writes_variable("i" + name);
    };
    bool writes_concentration_int() const {
        return writes_variable(name + "i");
    };
    bool writes_concentration_ext() const {
        return writes_variable(name + "o");
    };
    bool writes_rev_potential() const {
        return writes_variable("e" + name);
    };

    bool uses_valence() const {
        return valence_var.type == tok::identifier;
    }
    bool verifies_valence() const {
        return has_valence_expr && !uses_valence();
    }

    bool reads_variable(const std::string& name) const {
        return std::find_if(read.begin(), read.end(), [&name](const Token& t) { return t.spelling == name; }) != read.end();
    }
    bool writes_variable(const std::string& name) const {
        return std::find_if(write.begin(), write.end(), [&name](const Token& t) { return t.spelling == name; }) != write.end();
    }
};

typedef std::vector<Token> unit_tokens;
struct Id {
    Token token;
    std::string value; // store the value as a string, not a number : empty
                       // string == no value
    unit_tokens units;

    std::pair<std::string, std::string> range; // empty component => no range set

    Id(Token const& t, std::string const& v, unit_tokens const& u):
        token(t),
        value(v),
        units(u) {}

    Id() {}

    bool has_value() const {
        return !value.empty();
    }

    bool has_range() const {
        return !range.first.empty();
    }

    std::string unit_string() const {
        std::string u;
        for (auto& t: units) {
            if (!u.empty()) {
                u += ' ';
            }
            u += t.spelling;
        }
        return u;
    }

    std::string const& name() const {
        return token.spelling;
    }
};

// information stored in a NEURON {} block in mod file.
struct NeuronBlock {
    bool threadsafe = false;
    std::string name;
    moduleKind kind;
    std::vector<IonDep> ions;
    std::vector<Token> ranges;
    std::vector<Token> globals;
    Token nonspecific_current;
    bool has_nonspecific_current() const {
        return nonspecific_current.spelling.size() > 0;
    }
};

// information stored in a NEURON {} block in mod file
struct StateBlock {
    std::vector<Id> state_variables;
    auto begin() -> decltype(state_variables.begin()) {
        return state_variables.begin();
    }
    auto end() -> decltype(state_variables.end()) {
        return state_variables.end();
    }
};

// information stored in a NEURON {} block in mod file
struct UnitsBlock {
    typedef std::pair<unit_tokens, unit_tokens> units_pair;
    std::vector<units_pair> unit_aliases;
};

// information stored in a NEURON {} block in mod file
struct ParameterBlock {
    std::vector<Id> parameters;

    auto begin() -> decltype(parameters.begin()) {
        return parameters.begin();
    }
    auto end() -> decltype(parameters.end()) {
        return parameters.end();
    }
};

// information stored in a ASSIGNED {} block in mod file
struct AssignedBlock {
    std::vector<Id> parameters;

    auto begin() -> decltype(parameters.begin()) {
        return parameters.begin();
    }
    auto end() -> decltype(parameters.end()) {
        return parameters.end();
    }
};

// information stored in a WHITE_NOISE {} block in mod file
struct WhiteNoiseBlock {
    std::vector<Id> parameters;
    std::map<std::string, unsigned int> used;

    auto begin() -> decltype(parameters.begin()) {
        return parameters.begin();
    }
    auto end() -> decltype(parameters.end()) {
        return parameters.end();
    }
};

////////////////////////////////////////////////
// helpers for pretty printing block information
////////////////////////////////////////////////

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, Id const& V);

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, UnitsBlock::units_pair const& p);

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, IonDep const& I);

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, moduleKind const& k);

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, NeuronBlock const& N);

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, StateBlock const& B);

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, UnitsBlock const& U);

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, ParameterBlock const& P);

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, AssignedBlock const& A);

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, WhiteNoiseBlock const& W);
