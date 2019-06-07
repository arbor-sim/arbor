#pragma once

#include <stdexcept>
#include <string>

namespace pyarb {

// Python wrapper errors

struct pyarb_error: std::runtime_error {
    pyarb_error(const std::string& what_msg):
        std::runtime_error(what_msg) {}
    pyarb_error(const char* what_msg):
        std::runtime_error(what_msg) {}
};

inline
void assert_throw(bool pred, const char* msg) {
    if (!pred) throw pyarb_error(msg);
}

} // namespace pyarb
