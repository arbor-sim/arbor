#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "identifier.hpp"
#include "location.hpp"
#include "token.hpp"
#include "modccutil.hpp"

// describes a relationship with an ion channel
struct IonDep {
    ionKind kind() const {
        return to_ionKind(name);
    }
    std::string name;         // name of ion channel
    std::vector<Token> read;  // name of channels parameters to write
    std::vector<Token> write; // name of channels parameters to read

    bool has_variable(std::string const& name) {
        return writes_variable(name) || reads_variable(name);
    };
    bool uses_current() {
        return has_variable("i"+name);
    };
    bool uses_rev_potential() {
        return has_variable("e"+name);
    };
    bool uses_concentration_int() {
        return has_variable(name+"i");
    };
    bool uses_concentration_ext() {
        return has_variable(name+"o");
    };
    bool writes_concentration_int() {
        return writes_variable(name+"i");
    };
    bool writes_concentration_ext() {
        return writes_variable(name+"o");
    };

    bool reads_variable(const std::string& name) {
        return std::find_if(read.begin(), read.end(),
                [&name](const Token& t) {return t.spelling==name;}) != read.end();
    }
    bool writes_variable(const std::string& name) {
        return std::find_if(write.begin(), write.end(),
                [&name](const Token& t) {return t.spelling==name;}) != write.end();
    }
};

enum class moduleKind {
    point,
    density
};

typedef std::vector<Token> unit_tokens;
struct Id {
    Token token;
    std::string value; // store the value as a string, not a number : empty
                       // string == no value
    unit_tokens units;

    std::pair<Token, Token> range; // empty component => no range set

    Id(Token const& t, std::string const& v, unit_tokens const& u)
        : token(t), value(v), units(u)
    {}

    Id() {}

    bool has_value() const {
        return !value.empty();
    }

    bool has_range() const {
        return !range.first.spelling.empty();
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
        return nonspecific_current.spelling.size()>0;
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

// information stored in a NEURON {} block in mod file
struct AssignedBlock {
    std::vector<Id> parameters;

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
inline std::ostream& operator<< (std::ostream& os, Id const& V) {
    if(V.units.size())
        os << "(" << V.token << "," << V.value << "," << V.units << ")";
    else
        os << "(" << V.token << "," << V.value << ",)";

    return os;
}

inline std::ostream& operator<< (std::ostream& os, UnitsBlock::units_pair const& p) {
    return os << "(" << p.first << ", " << p.second << ")";
}

inline std::ostream& operator<< (std::ostream& os, IonDep const& I) {
    return os << "(" << I.name << ": read " << I.read << " write " << I.write << ")";
}

inline std::ostream& operator<< (std::ostream& os, moduleKind const& k) {
    return os << (k==moduleKind::density ? "density" : "point process");
}

inline std::ostream& operator<< (std::ostream& os, NeuronBlock const& N) {
    os << blue("NeuronBlock")     << std::endl;
    os << "  kind       : " << N.kind  << std::endl;
    os << "  name       : " << N.name  << std::endl;
    os << "  threadsafe : " << (N.threadsafe ? "yes" : "no") << std::endl;
    os << "  ranges     : " << N.ranges  << std::endl;
    os << "  globals    : " << N.globals << std::endl;
    os << "  ions       : " << N.ions    << std::endl;

    return os;
}

inline std::ostream& operator<< (std::ostream& os, StateBlock const& B) {
    os << blue("StateBlock")      << std::endl;
    return os << "  variables  : " << B.state_variables << std::endl;

}

inline std::ostream& operator<< (std::ostream& os, UnitsBlock const& U) {
    os << blue("UnitsBlock")      << std::endl;
    os << "  aliases    : "  << U.unit_aliases << std::endl;

    return os;
}

inline std::ostream& operator<< (std::ostream& os, ParameterBlock const& P) {
    os << blue("ParameterBlock")   << std::endl;
    os << "  parameters : "  << P.parameters << std::endl;

    return os;
}

inline std::ostream& operator<< (std::ostream& os, AssignedBlock const& A) {
    os << blue("AssignedBlock")   << std::endl;
    os << "  parameters : "  << A.parameters << std::endl;

    return os;
}
