#pragma once

#include <cstring>
#include <string>
#include <stdexcept>

enum class moduleKind {
    point, density, revpot
};

/// indicate how a variable is accessed
/// access is (read, written, or both)
/// the distinction between write only and read only is required because
/// if an external variable is to be written/updated, then it does not have
/// to be loaded before applying a kernel.
enum class accessKind {
    read,
    write,
    readwrite,
    noaccess
};

/// describes the scope of a variable
enum class visibilityKind {
    local,
    global
};

/// describes the scope of a variable
enum class rangeKind {
    range,
    scalar
};

/// the whether the variable value is defined inside or outside of the module.
enum class linkageKind {
    local,
    external
};

/// possible external data source for indexed variables
enum class sourceKind {
    voltage,
    current_density,
    current,
    conductivity,
    conductance,
    dt,
    ion_current,
    ion_current_density,
    ion_revpot,
    ion_iconc,
    ion_econc,
    ion_valence,
    temperature,
    diameter,
    no_source
};

inline std::string yesno(bool val) {
    return std::string(val ? "yes" : "no");
};

////////////////////////////////////////////
// to_string functions convert types
// to strings for printing diagnostics
////////////////////////////////////////////

inline std::string to_string(visibilityKind v) {
    switch(v) {
        case visibilityKind::local : return std::string("local");
        case visibilityKind::global: return std::string("global");
    }
    return std::string("<error : undefined visibilityKind>");
}

inline std::string to_string(linkageKind v) {
    switch(v) {
        case linkageKind::local : return std::string("local");
        case linkageKind::external: return std::string("external");
    }
    return std::string("<error : undefined visibilityKind>");
}

inline std::string to_string(sourceKind v) {
    switch(v) {
    case sourceKind::voltage:             return "voltage";
    case sourceKind::current_density:     return "current_density";
    case sourceKind::current:             return "current";
    case sourceKind::conductivity:        return "conductivity";
    case sourceKind::conductance:         return "conductance";
    case sourceKind::dt:                  return "dt";
    case sourceKind::ion_current:         return "ion_current";
    case sourceKind::ion_current_density: return "ion_current_density";
    case sourceKind::ion_revpot:          return "ion_revpot";
    case sourceKind::ion_iconc:           return "ion_iconc";
    case sourceKind::ion_econc:           return "ion_econc";
    case sourceKind::ion_valence:         return "ion_valence";
    case sourceKind::temperature:         return "temperature";
    case sourceKind::no_source:           return "no source";
    default:                              return "unknown source";
    }
}

// ostream writers

inline std::ostream& operator<< (std::ostream& os, visibilityKind v) {
    return os << to_string(v);
}

inline std::ostream& operator<< (std::ostream& os, linkageKind l) {
    return os << to_string(l);
}

/// ion variable to data source kind

inline sourceKind ion_source(const std::string& ion, const std::string& var, moduleKind mkind) {
    if (ion.empty()) return sourceKind::no_source;
    else if (var=="i"+ion) return mkind==moduleKind::point? sourceKind::ion_current: sourceKind::ion_current_density;
    else if (var=="e"+ion) return sourceKind::ion_revpot;
    else if (var==ion+"i") return sourceKind::ion_iconc;
    else if (var==ion+"o") return sourceKind::ion_econc;
    else return sourceKind::no_source;
}

