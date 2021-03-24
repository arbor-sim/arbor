#include <stdexcept>
#include <string>

#include <libxml/parser.h>

#include <arborio/xml.hpp>

// Implementations for exposed libxml2 interfaces.

namespace arborio {

xml_error::xml_error(const std::string& xml_error_msg, unsigned line):
    std::runtime_error(std::string("xml error: ") + (line? "line " + std::to_string(line): "") + xml_error_msg),
    xml_error_msg(xml_error_msg),
    line(line)
{}

with_xml::with_xml(): run_cleanup_(true) {
    // Initialize before any multithreaded access by library or client code.
    xmlInitParser();
}

with_xml::with_xml(with_xml&& other): run_cleanup_(other.run_cleanup_) {
    other.run_cleanup_ = false;
}

with_xml::~with_xml() {
    if (run_cleanup_) {
        xmlCleanupParser();
    }
}

} // namespace arborio
