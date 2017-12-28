#pragma once

#include <cxxabi.h>

/*
 * Utility code for generating string representations of types for use in Python
 * wrappers.
 */

#include <string>
#include <sstream>

#include <profiling/meter_manager.hpp>
#include <recipe.hpp>
#include <rss_cell.hpp>

inline
std::string demangle(const char* mangled) {
    int status;
    char *demangled = abi::__cxa_demangle(mangled, 0, 0, &status);
    std::string name;
    if (demangled!=nullptr) {
        name = demangled;
        std::free(demangled);
    }
    else {
        name = "<unknown type>";
    }
    return name;
}

inline
std::string any_string(const arb::util::any& a) {
    if (a.has_value()) {
        return std::string("[Arbor any: ") + demangle(a.type().name()) + "]";
    }
    return "[Arbor any: empty]";
}

inline
std::string meter_report_string(const arb::util::meter_report& r) {
    std::stringstream s;
    s << "Arbor meter report:\n";
    s << r;
    return s.str();
}

inline
std::string rss_cell_string(const arb::rss_cell& c) {
    return "[regular spiking cell: "
         "from " + std::to_string(c.start_time)
        +" to "  + std::to_string(c.stop_time)
        +" by "  + std::to_string(c.period) +"]";
}
