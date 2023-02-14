#pragma once

#include <arborio/export.hpp>

#include <string>

#include <nlohmann/json.hpp>

using key_type = std::string;

namespace arborio {

struct ARB_ARBORIO_API json_serdes {
    using json = nlohmann::json;

    template <typename V>
    void write(const key_type& k, const V& v) { data[ptr / std::string(k)] = v; }

    template <typename V>
    void read(const key_type& k, V& v) { data[ptr / std::string(k)].get_to(v); }

    std::optional<key_type> next_key() {
        if (iter.empty()) return {};
        auto& [it, end] = iter.back();
        if (it == end) return {};
        auto key = it.key();
        it++;
        return key;
    }

    void begin_write_map(const key_type& k) {
        ptr /= std::string(k);
        data[ptr] = json::object();  // NOTE we need this in case someone writes out an integer keyed map, lest we get an array!
    }
    void end_write_map() { ptr.pop_back(); }
    void begin_write_array(const key_type& k) {
        ptr /= std::string(k);
        data[ptr] = json::array(); // NOTE technically not needed, but gives nice output if empty
    }
    void end_write_array() { ptr.pop_back(); }

    void begin_read_map(const key_type& k) {
        ptr /= k;
        auto items = data[ptr].items();
        iter.push_back(range{items.begin(), items.end()});
    }
    void end_read_map() {
        ptr.pop_back();
        iter.pop_back();
    }
    void begin_read_array(const key_type& k) { begin_read_map(k); }
    void end_read_array() { end_read_map(); }

    const json& get_json() const { return data; }

private:
    json data;
    json::json_pointer ptr{""};
    struct range {
        decltype(data.items().begin()) begin;
        decltype(data.items().end()) end;
    };
    std::vector<range> iter;
};

}
