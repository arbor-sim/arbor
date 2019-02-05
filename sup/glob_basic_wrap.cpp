#include <string>
#include <vector>

#include <sup/glob.hpp>

namespace sup {

std::vector<path> glob(const std::string& pattern) {
    return glob_basic(pattern);
}

} // namespace sup

