#include <arbornml/with_xml.hpp>

extern "C" {
#include <libxml/parser.h>
}

namespace arbnml {

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

} // namespace arbnml
