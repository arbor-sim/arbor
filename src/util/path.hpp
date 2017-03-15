#pragma once

/*
 * Small subset of C++17 filesystem::path functionality
 *
 * Missing functionality:
 *   * Wide character methods
 *   * Locale-aware conversions
 *   * Path decomposition
 *   * Path element queries
 *   * Lexical normalizations and comparisons
 *   * Path element iteration
 * 
 * If we do want to support non-POSIX paths properly, consider
 * making posix::path into a generic `basic_path` parameterized
 * on `value_type`, `preferred_separator`, and using CRTP to
 * handle system-specific conversions.
 */

#include <string>
#include <iostream>
#include <utility>

#include <util/meta.hpp>
#include <util/rangeutil.hpp>

namespace nest {
namespace mc {
namespace util {

namespace posix {
    class path {
    public:
        using value_type = char;
        using string_type = std::basic_string<value_type>;
        static constexpr value_type preferred_separator = '/';

        path() = default;
        path(const path&) = default;
        path(path&&) = default;

        path& operator=(const path&) = default;
        path& operator=(path&&) = default;

        // Swap internals

        void swap(path& other) {
            p_.swap(other.p_);
        }

        // Construct or assign from value_type string or sequence.

        template <typename Source>
        path(Source&& source) { assign(std::forward<Source>(source)); }

        template <typename Iter>
        path(Iter b, Iter e) { assign(b, e); }

        template <typename Source>
        path& operator=(const Source& source) { return assign(source); }

        path& assign(const path& other) {
            p_ = other.p_;
            return *this;
        }

        path& assign(const string_type& source) {
            p_ = source;
            return *this;
        }

        path& assign(const value_type* source) {
            p_ = source;
            return *this;
        }

        template <typename Seq, typename = enable_if_sequence_t<Seq>>
        path& assign(const Seq& seq) {
            util::assign(p_, seq);
            return *this;
        }

        template <typename Iter>
        path& assign(Iter b, Iter e) {
            p_.assign(b, e);
            return *this;
        }

        // Empty?
        bool empty() const { return p_.empty(); }

        // Make empty
        void clear() { p_.clear(); }

        // Append path components

        template <typename Source>
        path& append(const Source& source) {
            return append(path(source));
        }

        template <typename Iter>
        path& append(Iter b, Iter e) {
            return append(path(b, e));
        }

        template <typename Source>
        path& operator/=(const Source& source) {
            return append(path(source));
        }

        path& append(const path& tail) {
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

        // Concat to path

        template <typename Source>
        path& concat(const Source& source) {
            return concat(path(source));
        }

        template <typename Iter>
        path& concat(Iter b, Iter e) {
            return concat(path(b, e));
        }

        template <typename Source>
        path& operator+=(const Source& source) {
           return  concat(path(source));
        }

        path& concat(const path& tail) {
            p_ += tail.p_;
            return *this;
        }

        // Native path string

        const value_type* c_str() const { return p_.c_str(); }
        const string_type& native() const { return p_; }
        operator string_type() const { return p_; }

        // Generic path string (same for POSIX)

        std::string generic_string() const { return p_; }

        // Queries

        bool is_absolute() const {
            return !p_.empty() && p_.front()==preferred_separator;
        }

        bool is_relative() const {
            return !is_absolute();
        }

        // Compare

        int compare(const string_type& s) const {
            return compare(path(s));
        }

        int compare(const value_type* s) const {
            return compare(path(s));
        }

        int compare(const path& other) const {
            // TODO: replace with cleaner implementation if/when iterators
            // are implemented.

            return canonical().compare(other.canonical());
        }

        // Non-member functions

        friend path operator/(const path& a, const path& b) {
            path p(a);
            return p/=b;
        }

        friend std::size_t hash_value(const path& p) {
            std::hash<path::string_type> hash;
            return hash(p.p_);
        }

        friend std::ostream& operator<<(std::ostream& o, const path& p) {
            return o << p.native();
        }

        friend std::istream& operator>>(std::istream& i, path& p) {
            path::string_type s;
            i >> s;
            p = s;
            return i;
        }

        friend void swap(path& lhs, path& rhs) {
            lhs.swap(rhs);
        }

        friend bool operator<(const path& p, const path& q) {
            return p.compare(q)<0;
        }

        friend bool operator>(const path& p, const path& q) {
            return p.compare(q)>0;
        }

        friend bool operator==(const path& p, const path& q) {
            return p.compare(q)==0;
        }

        friend bool operator!=(const path& p, const path& q) {
            return p.compare(q)!=0;
        }

        friend bool operator<=(const path& p, const path& q) {
            return p.compare(q)<=0;
        }

        friend bool operator>=(const path& p, const path& q) {
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
} // namespace posix

using path = posix::path;

// POSIX glob (3) wrapper.
std::vector<path> glob(const std::string& pattern);

} // namespace util
} // namespace mc
} // namespace nest

