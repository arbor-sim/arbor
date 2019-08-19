#pragma once

#include <stdexcept>

#include <arbor/arbexcept.hpp>

namespace arb {

struct morphology_error: public arbor_exception {
    morphology_error(const char* what): arbor_exception(what) {}
    morphology_error(const std::string& what): arbor_exception(what) {}
};

} // namespace arb

