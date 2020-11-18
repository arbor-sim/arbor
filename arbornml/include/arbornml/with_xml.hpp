#pragma once

// Wrap initialization and cleanup of libxml2 library.
//
// Use of `with_xml` is only necessary if arbornml is being
// used in a multithreaded context and the client code is
// not managing libxml2 initialization and cleanup.

namespace arbnml {

struct with_xml {
    with_xml();
    ~with_xml();

    with_xml(with_xml&&);
    with_xml(const with_xml&) = delete;

    with_xml& operator=(const with_xml&) = delete;
    with_xml& operator=(with_xml&&) = delete;

    bool run_cleanup_;
};

} // namespace arbnml
