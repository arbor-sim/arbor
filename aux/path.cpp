// POSIX headers
extern "C" {
#include <sys/stat.h>
}

#include <cerrno>

#include <aux/path.hpp>

namespace aux {
namespace posix {

namespace impl {
    file_status status(const char* p, int r, struct stat& st, std::error_code& ec) {
        if (!r) {
            // Success:
            ec.clear();
            perms p = static_cast<perms>(st.st_mode&07777);
            switch (st.st_mode&S_IFMT) {
            case S_IFSOCK:
                return file_status{file_type::socket, p};
            case S_IFLNK:
                return file_status{file_type::symlink, p};
            case S_IFREG:
                return file_status{file_type::regular, p};
            case S_IFBLK:
                return file_status{file_type::block, p};
            case S_IFDIR:
                return file_status{file_type::directory, p};
            case S_IFCHR:
                return file_status{file_type::character, p};
            case S_IFIFO:
                return file_status{file_type::fifo, p};
            default:
                return file_status{file_type::unknown, p};
            }
        }

        // Handle error cases especially.

        if ((errno==ENOENT || errno==ENOTDIR) && p && *p) {
            // If a non-empty path, return `not_found`.
            ec.clear();
            return file_status{file_type::not_found};
        }
        else {
            ec = std::error_code(errno, std::generic_category());
            return file_status{file_type::none};
        }
    }
} // namespace impl


file_status status(const path& p, std::error_code& ec) {
    struct stat st;
    int r = stat(p.c_str(), &st);
    return impl::status(p.c_str(), r, st, ec);
}

file_status symlink_status(const path& p, std::error_code& ec) {
    struct stat st;
    int r = lstat(p.c_str(), &st);
    return impl::status(p.c_str(), r, st, ec);
}

} // namespace posix
} // namespace arb

