#include <fstream>
#include <utility>

#include <aux/ioutil.hpp>
#include <aux/path.hpp>
#include <aux/strsub.hpp>

namespace aux {

std::fstream open_or_throw(const path& p, std::ios_base::openmode mode, bool exclusive) {
    if (exclusive && exists(p)) {
        throw std::runtime_error(strsub("file % already exists", p));
    }

    std::fstream file;
    file.open(p, mode);
    if (!file) {
        throw std::runtime_error(strsub("unable to open file %", p));
    }

    return std::move(file);
}

} // namespace aux

