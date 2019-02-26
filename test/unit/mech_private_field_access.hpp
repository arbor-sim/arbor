#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arbor/mechanism.hpp>

// Get a copy of the data within a mechanisms's (private) named field.

std::vector<arb::fvm_value_type> mechanism_field(arb::mechanism* m, const std::string& key);

template <typename DerivedMechPtr>
inline std::vector<arb::fvm_value_type> mechanism_field(const std::unique_ptr<DerivedMechPtr>& m, const std::string& key) {
    return mechanism_field(static_cast<arb::mechanism*>(m.get()), key);
}
