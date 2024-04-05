#include <fstream>
#include <utility>

#include <fmt/format.h>

#include <sup/ioutil.hpp>

namespace sup {

ARB_SUP_API std::fstream open_or_throw(const std::filesystem::path& p, std::ios_base::openmode mode, bool exclusive) {
    if (exclusive && std::filesystem::exists(p)) throw std::runtime_error(fmt::format("file {} already exists", p.string()));
    std::fstream file;
    file.open(p, mode);
    if (!file) throw std::runtime_error(fmt::format("unable to open file {}", p.string()));
    return file;
}

} // namespace sup

