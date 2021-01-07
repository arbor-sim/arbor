#pragma once

#include <exception>
#include <optional>

#include <nlohmann/json.hpp>

// Search a json object for an entry with a given name.
// If found, return the value and remove from json object.
template <typename T>
std::optional<T> find_and_remove_json(const char* name, nlohmann::json& j) {
    auto it = j.find(name);
    if (it==j.end()) {
        throw std::runtime_error("parameter "+std::string(name)+" not found");
    }
    T value = std::move(*it);
    j.erase(name);
    return std::move(value);
}