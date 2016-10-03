#pragma once

#include <string>

/// indicate how a variable is accessed
/// access is (read, written, or both)
/// the distinction between write only and read only is required because
/// if an external variable is to be written/updated, then it does not have
/// to be loaded before applying a kernel.
enum class accessKind {
    read,
    write,
    readwrite
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

/// ion channel that the variable belongs to
enum class ionKind {
    none,     ///< not an ion variable
    nonspecific,  ///< nonspecific current
    Ca,       ///< calcium ion
    Na,       ///< sodium ion
    K         ///< potassium ion
};

static std::string yesno(bool val) {
    return std::string(val ? "yes" : "no");
};

////////////////////////////////////////////
// to_string functions convert types
// to strings for printing diagnostics
////////////////////////////////////////////
static std::string to_string(ionKind i) {
    switch(i) {
        case ionKind::none : return std::string("none");
        case ionKind::Ca   : return std::string("calcium");
        case ionKind::Na   : return std::string("sodium");
        case ionKind::K    : return std::string("potassium");
        case ionKind::nonspecific : return std::string("nonspecific");
    }
    return std::string("<error : undefined ionKind>");
}

static std::string to_string(visibilityKind v) {
    switch(v) {
        case visibilityKind::local : return std::string("local");
        case visibilityKind::global: return std::string("global");
    }
    return std::string("<error : undefined visibilityKind>");
}

static std::string to_string(linkageKind v) {
    switch(v) {
        case linkageKind::local : return std::string("local");
        case linkageKind::external: return std::string("external");
    }
    return std::string("<error : undefined visibilityKind>");
}

// ostream writers
static std::ostream& operator<< (std::ostream& os, ionKind i) {
    return os << to_string(i);
}

static std::ostream& operator<< (std::ostream& os, visibilityKind v) {
    return os << to_string(v);
}

static std::ostream& operator<< (std::ostream& os, linkageKind l) {
    return os << to_string(l);
}

static ionKind ion_kind_from_name(std::string field) {
    if(field.substr(0,4) == "ion_") {
        field = field.substr(4);
    }
    if(field=="ica" || field=="eca" || field=="cai" || field=="cao") {
        return ionKind::Ca;
    }
    if(field=="ik" || field=="ek" || field=="ki" || field=="ko") {
        return ionKind::K;
    }
    if(field=="ina" || field=="ena" || field=="nai" || field=="nao") {
        return ionKind::Na;
    }
    return ionKind::none;
}

static std::string ion_store(ionKind k) {
    switch(k) {
        case ionKind::Ca:
            return "ion_ca";
        case ionKind::Na:
            return "ion_na";
        case ionKind::K:
            return "ion_k";
        default:
            return "";
    }
}
