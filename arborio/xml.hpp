#pragma once

#include <arborio/neuroml.hpp>

#include <iostream>
#include <string>
#include <optional>

#include <pugixml.hpp>

namespace arborio {

using xml_node = pugi::xml_node;
using xml_doc  = pugi::xml_document;

template<typename T>
T get_attr(const xml_node& n,
           const std::string& a,
           std::optional<T> d={}) {
    auto attr = n.attribute(a.data());
    if (attr.empty()) {
        if (!d) throw nml_parse_error("Required attribute " + a + " is empty/absent.");
        return *d;
    }
    std::string val = attr.value();
    if constexpr (std::is_same_v<T, double>) {
        return std::stod(val);
    }
    if constexpr (std::is_same_v<T, std::string>) {
        return val;
    }
    if constexpr (std::is_unsigned_v<T>) {
        std::size_t n = 0;
        long long int i = std::stoull(val, &n);
        // Either we didn't consume all chars -- eg 1.6 -- or we consumed all, but the result is negative.
        // This _should_ be considered a bug in std::toull...
        if (n != val.size() || i < 0) throw nml_parse_error("Couldn't parse unsigned integer: " + val);
        return i;
    }
    throw std::runtime_error{"Not implemented"};
}

inline
std::string xpath_escape(const std::string& x) {
    auto npos = std::string::npos;
     if (x.find_first_of("'")==npos) {
         return "'"+x+"'";
     }
     else if (x.find_first_of("\"")==npos) {
         return "\""+x+"\"";
     }
     else {
         std::string r = "concat(";
         std::string::size_type i = 0;
         for (;;) {
             auto j = x.find_first_of("'", i);
             r += "'";
             r.append(x, i, j==npos? j: j-i);
             r += "'";
             if (j==npos) break;
             r += ",\"";
             i = j+1;
             j = x.find_first_not_of("'",i);
             r.append(x, i, j==npos? j: j-i);
             r += "\"";
             if (j==npos) break;
             r += ",";
             i = j+1;
         }
         r += ")";
         return r;
     }
}

}
