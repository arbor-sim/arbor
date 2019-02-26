#include <string>

#include "exception.hpp"

namespace pyarb {

python_error::python_error(const std::string& message):
    arbor_exception("arbor python wrapper error: " + message + "\n")
{}

} // namespace pyarb
