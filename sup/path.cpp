// POSIX headers
extern "C" {
#define _DEFAULT_SOURCE
#include <sys/stat.h>
#include <dirent.h>
}

#include <cerrno>

#include <sup/path.hpp>

namespace sup {

namespace impl {
    file_status status(const char* p, int r, struct stat& st, std::error_code& ec) noexcept {
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


file_status posix_status(const path& p, std::error_code& ec) noexcept {
    struct stat st;
    int r = stat(p.c_str(), &st);
    return impl::status(p.c_str(), r, st, ec);
}

file_status posix_symlink_status(const path& p, std::error_code& ec) noexcept {
    struct stat st;
    int r = lstat(p.c_str(), &st);
    return impl::status(p.c_str(), r, st, ec);
}

struct posix_directory_state {
    DIR* dir = nullptr;
    path dir_path;
    directory_entry entry;

    posix_directory_state() = default;
    ~posix_directory_state() {
        if (dir) closedir(dir);
    }
};

posix_directory_iterator::posix_directory_iterator(const path& p, directory_options diropt) {
    std::error_code ec;
    *this = posix_directory_iterator(p, diropt, ec);
    if (ec) throw filesystem_error("opendir()", p, ec);
}

posix_directory_iterator::posix_directory_iterator(const path& p, directory_options diropt, std::error_code& ec):
    state_(new posix_directory_state())
{
    ec.clear();
    if ((state_->dir = opendir(p.c_str()))) {
        state_->dir_path = p;
        increment(ec);
        return;
    }

    if (errno==EACCES && (diropt&directory_options::skip_permission_denied)!=0) return;
    ec = std::error_code(errno, std::generic_category());
}

static inline bool is_dot_or_dotdot(const char* s) {
    return *s=='.' && (!s[1] || (s[1]=='.' && !s[2]));
}

posix_directory_iterator& posix_directory_iterator::increment(std::error_code &ec) {
    enum file_type type = file_type::none;

    ec.clear();
    if (!state_->dir) return *this;

    struct dirent* dp = nullptr;
    do {
        errno = 0;
        dp = readdir(state_->dir);
    } while (dp && is_dot_or_dotdot(dp->d_name));

    if (!dp) {
        if (errno) ec = std::error_code(errno, std::generic_category());
        state_.reset();
    }
    else {
#if defined(DT_UNKNOWN)
        switch (dp->d_type) {
        case DT_BLK:
            type = file_type::block;
            break;
        case DT_CHR:
            type = file_type::character;
            break;
        case DT_DIR:
            type = file_type::directory;
            break;
        case DT_FIFO:
            type = file_type::fifo;
            break;
        case DT_LNK:
            type = file_type::symlink;
            break;
        case DT_SOCK:
            type = file_type::socket;
            break;
        case DT_UNKNOWN: // fallthrough
        default:
            type = file_type::unknown;
        }
#else
        type = file_type::unknown;
#endif
        state_->entry = directory_entry(state_->dir_path/path(dp->d_name), type, ec);
    }

    return *this;
}

posix_directory_iterator& posix_directory_iterator::operator++() {
    std::error_code ec;
    increment(ec);
    if (ec) throw filesystem_error("readdir()", state_->dir_path, ec);
    return *this;
}

directory_entry posix_directory_iterator::operator*() const {
    return state_->entry;
}

const directory_entry* posix_directory_iterator::operator->() const {
    return &(state_->entry);
}

bool posix_directory_iterator::operator==(const posix_directory_iterator& x) const {
    bool end1 = !state_ || !state_->dir;
    bool end2 = !x.state_ || !x.state_->dir;
    return end1 || end2? end1 && end2: state_->entry == x.state_->entry;
}

} // namespace sup

