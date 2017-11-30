#pragma once

#include <expression.hpp>
#include <identifier.hpp>

// Holds the state required to generate a write_back call in a mechanism.
struct WriteBack {
    // Name of the symbol inside the mechanism used to store.
    //      must be a state field
    std::string source_name;
    // Name of the field in the ion channel being written to.
    std::string target_name;
    // The ion channel being written to.
    //      must not be ionKind::none
    ionKind ion_kind;

    WriteBack(std::string src, std::string tgt, ionKind k):
        source_name(std::move(src)), target_name(std::move(tgt)), ion_kind(k)
    {}
};

