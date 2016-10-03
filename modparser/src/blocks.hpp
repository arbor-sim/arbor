#pragma once

#include <string>
#include <vector>

#include "identifier.hpp"
#include "location.hpp"
#include "token.hpp"
#include "util.hpp"

// describes a relationship with an ion channel
struct IonDep {
    ionKind kind() const {
        if(name=="k")  return ionKind::K;
        if(name=="na") return ionKind::Na;
        if(name=="ca") return ionKind::Ca;
        return ionKind::none;
    }
    std::string name;         // name of ion channel
    std::vector<Token> read;  // name of channels parameters to write
    std::vector<Token> write; // name of channels parameters to read
};

enum class moduleKind {
    point,
    density
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
    std::vector<std::string> state_variables;
    auto begin() -> decltype(state_variables.begin()) {
        return state_variables.begin();
    }
    auto end() -> decltype(state_variables.end()) {
        return state_variables.end();
    }
};

// information stored in a NEURON {} block in mod file
typedef std::vector<Token> unit_tokens;
struct UnitsBlock {
    typedef std::pair<unit_tokens, unit_tokens> units_pair;
    std::vector<units_pair> unit_aliases;
};

struct Id {
    Token token;
    std::string value; // store the value as a string, not a number : empty string == no value
    unit_tokens units;

    Id(Token const& t, std::string const& v, unit_tokens const& u)
        : token(t), value(v), units(u)
    {}

    Id() {}

    bool has_value() const {
        return value.size()>0;
    }

    std::string const& name() const {
        return token.spelling;
    }
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
static std::ostream& operator<< (std::ostream& os, Id const& V) {
    if(V.units.size())
        os << "(" << V.token << "," << V.value << "," << V.units << ")";
    else
        os << "(" << V.token << "," << V.value << ",)";

    return os;
}

static std::ostream& operator<< (std::ostream& os, UnitsBlock::units_pair const& p) {
    return os << "(" << p.first << ", " << p.second << ")";
}

static std::ostream& operator<< (std::ostream& os, IonDep const& I) {
    return os << "(" << I.name << ": read " << I.read << " write " << I.write << ")";
}

static std::ostream& operator<< (std::ostream& os, moduleKind const& k) {
    return os << (k==moduleKind::density ? "density" : "point process");
}

static std::ostream& operator<< (std::ostream& os, NeuronBlock const& N) {
    os << blue("NeuronBlock")     << std::endl;
    os << "  kind       : " << N.kind  << std::endl;
    os << "  name       : " << N.name  << std::endl;
    os << "  threadsafe : " << (N.threadsafe ? "yes" : "no") << std::endl;
    os << "  ranges     : " << N.ranges  << std::endl;
    os << "  globals    : " << N.globals << std::endl;
    os << "  ions       : " << N.ions    << std::endl;

    return os;
}

static std::ostream& operator<< (std::ostream& os, StateBlock const& B) {
    os << blue("StateBlock")      << std::endl;
    return os << "  variables  : " << B.state_variables << std::endl;

}

static std::ostream& operator<< (std::ostream& os, UnitsBlock const& U) {
    os << blue("UnitsBlock")      << std::endl;
    os << "  aliases    : "  << U.unit_aliases << std::endl;

    return os;
}

static std::ostream& operator<< (std::ostream& os, ParameterBlock const& P) {
    os << blue("ParameterBlock")   << std::endl;
    os << "  parameters : "  << P.parameters << std::endl;

    return os;
}

static std::ostream& operator<< (std::ostream& os, AssignedBlock const& A) {
    os << blue("AssignedBlock")   << std::endl;
    os << "  parameters : "  << A.parameters << std::endl;

    return os;
}

