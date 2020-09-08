#pragma once

#include <array>
#include <exception>

#include <arbor/util/optional.hpp>

#include <nlohmann/json.hpp>

namespace sup {

// Search a json object for an entry with a given name.
// If found, return the value and remove from json object.
template <typename T>
arb::util::optional<T> find_and_remove_json(const char* name, nlohmann::json& j) {
    auto it = j.find(name);
    if (it==j.end()) {
        return arb::util::nullopt;
    }
    T value = std::move(*it);
    j.erase(name);
    return std::move(value);
}

template <typename T>
void param_from_json(T& x, const char* name, nlohmann::json& j) {
    if (auto o = find_and_remove_json<T>(name, j)) {
        x = *o;
    }
}

template <typename T, size_t N>
void param_from_json(std::array<T, N>& x, const char* name, nlohmann::json& j) {
    std::vector<T> y;
    if (auto o = find_and_remove_json<std::vector<T>>(name, j)) {
        y = *o;
        if (y.size()!=N) {
            throw std::runtime_error("parameter "+std::string(name)+" requires "+std::to_string(N)+" values");
        }
        std::copy(y.begin(), y.end(), x.begin());
    }
}

} // namespace sup
