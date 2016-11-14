#pragma once

#include <string>
#include <fstream>

namespace nest {
namespace mc {
namespace util {

inline bool file_exists(const std::string& file_path) {
    std::ifstream fid(file_path);
    return fid.good();
}

} // namespace util
} // namespace mc
} // namespace nest
