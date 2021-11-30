#pragma once

#include <stdexcept>
#include <string>

// XML related interfaces deriving from the underlying XML implementation library.

namespace arborio {

// Generic XML error (as reported by libxml2).
struct xml_error: std::runtime_error {
    xml_error(const std::string& xml_error_msg, unsigned line = 0);
    std::string xml_error_msg;
    unsigned line;
};

// Wrap initialization and cleanup of libxml2 library.
//
// Use of `with_xml` is only necessary if arborio is being
// used in a multithreaded context and the client code is
// not managing libxml2 initialization and cleanup.

struct with_xml {
    with_xml();
    ~with_xml();

    with_xml(with_xml&&);
    with_xml(const with_xml&) = delete;

    with_xml& operator=(const with_xml&) = delete;
    with_xml& operator=(with_xml&&) = delete;

    bool run_cleanup_;
};

} // namespace arborio
