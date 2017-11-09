#pragma once

/*
 * Small subset of C++17 filesystem::path functionality
 *
 * Missing path functionality:
 *   * Wide character methods
 *   * Locale-aware conversions
 *   * Path decomposition
 *   * Path element queries
 *   * Lexical normalizations and comparisons
 *   * Path element iteration
 * 
 * If we do want to support non-POSIX paths properly, consider
 * making posix_path into a generic `basic_path` parameterized
 * on `value_type`, `preferred_separator`, and using CRTP to
 * handle system-specific conversions.
 */

#include <cstddef>
#include <exception>
#include <string>
#include <iostream>
#include <utility>

#include <util/meta.hpp>
#include <util/rangeutil.hpp>

namespace arb {
namespace util {

class posix_path {
public:
    using value_type = char;
    using string_type = std::basic_string<value_type>;
    static constexpr value_type preferred_separator = '/';

    posix_path() = default;
    posix_path(const posix_path&) = default;
    posix_path(posix_path&&) = default;

    posix_path& operator=(const posix_path&) = default;
    posix_path& operator=(posix_path&&) = default;

    // Swap internals

    void swap(posix_path& other) {
        p_.swap(other.p_);
    }

    // Construct or assign from value_type string or sequence.

    template <typename Source>
    posix_path(Source&& source) { assign(std::forward<Source>(source)); }

    template <typename Iter>
    posix_path(Iter b, Iter e) { assign(b, e); }

    template <typename Source>
    posix_path& operator=(const Source& source) { return assign(source); }

    posix_path& assign(const posix_path& other) {
        p_ = other.p_;
        return *this;
    }

    posix_path& assign(const string_type& source) {
        p_ = source;
        return *this;
    }

    posix_path& assign(const value_type* source) {
        p_ = source;
        return *this;
    }

    template <typename Seq, typename = enable_if_sequence_t<Seq>>
    posix_path& assign(const Seq& seq) {
        util::assign(p_, seq);
        return *this;
    }

    template <typename Iter>
    posix_path& assign(Iter b, Iter e) {
        p_.assign(b, e);
        return *this;
    }

    bool empty() const { return p_.empty(); }

    void clear() { p_.clear(); }

    // Append posix_path components
    template <typename Source>
    posix_path& append(const Source& source) {
        return append(posix_path(source));
    }

    template <typename Iter>
    posix_path& append(Iter b, Iter e) {
        return append(posix_path(b, e));
    }

    template <typename Source>
    posix_path& operator/=(const Source& source) {
        return append(posix_path(source));
    }

    posix_path& append(const posix_path& tail) {
        if (!p_.empty() &&
            !is_separator(p_.back()) &&
            !tail.p_.empty() &&
            !is_separator(tail.p_.front()))
        {
            p_ += preferred_separator;
        }
        if (!p_.empty() &&
            !tail.empty() &&
            is_separator(p_.back()) &&
            is_separator(tail.p_.front()))
        {
            p_ += tail.p_.substr(1);
        }
        else {
            p_ += tail.p_;
        }
        return *this;
    }

    // Concat to posix_path
    template <typename Source>
    posix_path& concat(const Source& source) {
        return concat(posix_path(source));
    }

    template <typename Iter>
    posix_path& concat(Iter b, Iter e) {
        return concat(posix_path(b, e));
    }

    template <typename Source>
    posix_path& operator+=(const Source& source) {
       return  concat(posix_path(source));
    }

    posix_path& concat(const posix_path& tail) {
        p_ += tail.p_;
        return *this;
    }

    // Native posix_path string
    const value_type* c_str() const { return p_.c_str(); }
    const string_type& native() const { return p_; }
    operator string_type() const { return p_; }

    // Generic posix_path string (same for POSIX)
    std::string generic_string() const { return p_; }

    // Queries
    bool is_absolute() const {
        return !p_.empty() && p_.front()==preferred_separator;
    }

    bool is_relative() const {
        return !is_absolute();
    }

    int compare(const string_type& s) const {
        return compare(posix_path(s));
    }

    int compare(const value_type* s) const {
        return compare(posix_path(s));
    }

    int compare(const posix_path& other) const {
        // TODO: replace with cleaner implementation if/when iterators
        // are implemented.

        return canonical().compare(other.canonical());
    }

    // Non-member functions

    friend posix_path operator/(const posix_path& a, const posix_path& b) {
        posix_path p(a);
        return p/=b;
    }

    friend std::size_t hash_value(const posix_path& p) {
        std::hash<posix_path::string_type> hash;
        return hash(p.p_);
    }

    friend std::ostream& operator<<(std::ostream& o, const posix_path& p) {
        return o << p.native();
    }

    friend std::istream& operator>>(std::istream& i, posix_path& p) {
        posix_path::string_type s;
        i >> s;
        p = s;
        return i;
    }

    friend void swap(posix_path& lhs, posix_path& rhs) {
        lhs.swap(rhs);
    }

    friend bool operator<(const posix_path& p, const posix_path& q) {
        return p.compare(q)<0;
    }

    friend bool operator>(const posix_path& p, const posix_path& q) {
        return p.compare(q)>0;
    }

    friend bool operator==(const posix_path& p, const posix_path& q) {
        return p.compare(q)==0;
    }

    friend bool operator!=(const posix_path& p, const posix_path& q) {
        return p.compare(q)!=0;
    }

    friend bool operator<=(const posix_path& p, const posix_path& q) {
        return p.compare(q)<=0;
    }

    friend bool operator>=(const posix_path& p, const posix_path& q) {
        return p.compare(q)>=0;
    }

protected:
    static bool is_separator(value_type c) {
        return c=='/' || c==preferred_separator;
    }

    std::string canonical() const {
        std::string n;
        value_type prev = 0;
        for (value_type c: p_) {
            if (is_separator(c)) {
                if (!is_separator(prev)) {
                    n += '/';
                }
            }
            else {
                n += c;
            }
            prev = c;
        }
        if (!n.empty() && n.back()=='/') {
            n += '.';
        }
        return n;
    }

    string_type p_;
};

// Paths for now are just POSIX paths.
using path = posix_path;

// Following C++17 std::filesystem::perms:
enum class perms: unsigned {
    none = 0,
    owner_read = 0400,
    owner_write = 0200,
    owner_exec = 0100,
    owner_all = 0700,
    group_read = 040,
    group_write = 020,
    group_exec = 010,
    group_all = 070,
    others_read = 04,
    others_write = 02,
    others_exec = 01,
    others_all = 07,
    all = 0777,
    set_uid = 04000,
    set_gid = 02000,
    sticky_bit = 01000,
    mask = 07777,
    unknown = 0xffff,
    add_perms = 0x10000,
    remove_perms = 0x20000,
    resolve_symlinks = 0x40000,
};

inline perms operator&(perms x, perms y) { return static_cast<perms>(static_cast<unsigned>(x)&static_cast<unsigned>(y)); }
inline perms operator|(perms x, perms y) { return static_cast<perms>(static_cast<unsigned>(x)|static_cast<unsigned>(y)); }
inline perms operator^(perms x, perms y) { return static_cast<perms>(static_cast<unsigned>(x)^static_cast<unsigned>(y)); }
inline perms operator~(perms x) { return static_cast<perms>(~static_cast<unsigned>(x)); }
inline perms& operator&=(perms& x, perms y) { return x = x&y; }
inline perms& operator|=(perms& x, perms y) { return x = x|y; }
inline perms& operator^=(perms& x, perms y) { return x = x^y; }

enum class file_type {
    none, not_found, regular, directory, symlink, block,
    character, fifo, socket, unknown
};

class file_status {
public:
    explicit file_status(file_type type, perms permissions = perms::unknown):
        type_(type), perm_(permissions)
    {}

    file_status(): file_status(file_type::none) {}

    file_status(const file_status&) = default;
    file_status(file_status&&) = default;
    file_status& operator=(const file_status&) = default;
    file_status& operator=(file_status&&) = default;

    file_type type() const { return type_; }
    void type(file_type type) { type_ = type; }

    perms permissions() const { return perm_; }
    void permissions(perms perm) { perm_ = perm; }

private:
    file_type type_;
    perms perm_;
};

class filesystem_error: public std::system_error {
public:
    filesystem_error(const std::string& what_arg, std::error_code ec):
        std::system_error(ec, what_arg) {}

    filesystem_error(const std::string& what_arg, const path& p1, std::error_code ec):
        std::system_error(ec, what_arg), p1_(p1) {}

    filesystem_error(const std::string& what_arg, const path& p1, const path& p2, std::error_code ec):
        std::system_error(ec, what_arg), p1_(p1), p2_(p2) {}

    const path& path1() const { return p1_; }
    const path& path2() const { return p2_; }

private:
    path p1_, p2_;
};

// POSIX implementations of path queries (see path.cpp for implementations).

namespace posix {
    file_status status(const path&, std::error_code&);
    file_status symlink_status(const path&, std::error_code&);

    // POSIX glob (3) wrapper (not part of std::filesystem!).
    std::vector<path> glob(const std::string& pattern);
}

inline file_status status(const path& p, std::error_code& ec) {
    return posix::status(p, ec);
}

inline file_status symlink_status(const path& p, std::error_code& ec) {
    return posix::symlink_status(p, ec);
}

inline std::vector<path> glob(const std::string& pattern) {
    return posix::glob(pattern);
}

// Wrappers for `status()`, again following std::filesystem.

inline file_status status(const path& p) {
    std::error_code ec;
    auto r = ::arb::util::posix::status(p, ec);
    if (ec) {
        throw filesystem_error("status()", p, ec);
    }
    return r;
}

inline bool is_directory(file_status s) {
    return s.type()==file_type::directory;
}

inline bool is_directory(const path& p) {
    return is_directory(status(p));
}

inline bool is_directory(const path& p, std::error_code& ec) {
    return is_directory(status(p, ec));
}

inline bool is_regular_file(file_status s) {
    return s.type()==file_type::regular;
}

inline bool is_regular_file(const path& p) {
    return is_regular_file(status(p));
}

inline bool is_regular_file(const path& p, std::error_code& ec) {
    return is_regular_file(status(p, ec));
}

inline bool is_character_file(file_status s) {
    return s.type()==file_type::character;
}

inline bool is_character_file(const path& p) {
    return is_character_file(status(p));
}

inline bool is_character_file(const path& p, std::error_code& ec) {
    return is_character_file(status(p, ec));
}

inline bool exists(file_status s) {
    return s.type()!=file_type::not_found;
}

inline bool exists(const path& p) {
    return exists(status(p));
}

inline bool exists(const path& p, std::error_code& ec) {
    return exists(status(p, ec));
}

} // namespace util
} // namespace arb

