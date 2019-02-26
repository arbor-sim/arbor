#include "../gtest.h"

#include <sup/glob.hpp>
#include <sup/path.hpp>

using namespace sup;

#include <iterator>
#include <string>
#include <unordered_map>

TEST(glob, pattern) {
    EXPECT_TRUE( glob_basic_match( "literal", "literal"));
    EXPECT_FALSE(glob_basic_match("literal", "Literal"));

    EXPECT_TRUE( glob_basic_match("[a-z][A-Z]", "aA"));
    EXPECT_TRUE( glob_basic_match("[a-z][A-Z]", "zZ"));
    EXPECT_TRUE( glob_basic_match("[a-z][A-Z]", "bQ"));
    EXPECT_FALSE(glob_basic_match("[a-z][A-Z]", "AA"));
    EXPECT_FALSE(glob_basic_match("[a-z][A-Z]", "A@"));

    EXPECT_TRUE (glob_basic_match("[!0-9a]", "A"));
    EXPECT_FALSE(glob_basic_match("[!0-9a]", "0"));
    EXPECT_FALSE(glob_basic_match("[!0-9a]", "5"));
    EXPECT_FALSE(glob_basic_match("[!0-9a]", "9"));
    EXPECT_FALSE(glob_basic_match("[!0-9a]", "a"));

    EXPECT_TRUE (glob_basic_match("[-q]", "-"));
    EXPECT_TRUE (glob_basic_match("[-q]", "q"));
    EXPECT_FALSE(glob_basic_match("[-q]", "p"));

    EXPECT_TRUE (glob_basic_match("[q-]", "-"));
    EXPECT_TRUE (glob_basic_match("[q-]", "q"));
    EXPECT_FALSE(glob_basic_match("[-q]", "p"));

    EXPECT_TRUE (glob_basic_match("[!a-]", "b"));
    EXPECT_FALSE(glob_basic_match("[!a-]", "a"));
    EXPECT_FALSE(glob_basic_match("[!a-]", "-"));

    EXPECT_TRUE (glob_basic_match("[]-]z", "-z"));
    EXPECT_TRUE (glob_basic_match("[]-]z", "]z"));
    EXPECT_FALSE(glob_basic_match("[]-]z", "[z"));

    EXPECT_TRUE( glob_basic_match("?", "a"));
    EXPECT_TRUE( glob_basic_match("?", " "));
    EXPECT_FALSE(glob_basic_match("?", " a"));
    EXPECT_FALSE(glob_basic_match("?", ""));

    EXPECT_TRUE( glob_basic_match("a*b", "ab"));
    EXPECT_TRUE( glob_basic_match("a*b", "abb"));
    EXPECT_TRUE( glob_basic_match("a*b", "a01234b"));
    EXPECT_FALSE(glob_basic_match("a*b", "ac"));
    EXPECT_FALSE(glob_basic_match("a*b", "cb"));

    EXPECT_TRUE( glob_basic_match("a****b", "ab"));
    EXPECT_TRUE( glob_basic_match("a****b", "a01b"));
    EXPECT_FALSE(glob_basic_match("a****b", "a01"));

    EXPECT_TRUE( glob_basic_match("\\*", "*"));
    EXPECT_FALSE(glob_basic_match("\\*", "z"));

    EXPECT_TRUE( glob_basic_match("\\?", "?"));
    EXPECT_FALSE(glob_basic_match("\\?", "z"));

    EXPECT_TRUE( glob_basic_match("\\[p-q]", "[p-q]"));
    EXPECT_FALSE(glob_basic_match("\\[p-q]", "\\p"));
    EXPECT_TRUE( glob_basic_match("\\\\[p-q]", "\\p"));

    // Check for dodgy exponential behaviour...
    EXPECT_FALSE( glob_basic_match(
        "*x*x*x*x*x*x*x*x*x*x*x*x*x*x_",
        "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"));

    // Check special-case handling for initial period:

    EXPECT_FALSE(glob_basic_match("*",  ".foo"));
    EXPECT_TRUE( glob_basic_match(".*", ".foo"));

    EXPECT_FALSE(glob_basic_match("??",  ".f"));
    EXPECT_TRUE( glob_basic_match(".?",  ".f"));

    EXPECT_FALSE(glob_basic_match("[.a][.a][.a]", "..a"));
    EXPECT_TRUE( glob_basic_match(".[.a][.a]",  "..a"));

    EXPECT_TRUE( glob_basic_match("\\.*", ".foo"));
}

struct mock_fs_provider {
    using action_type = glob_fs_provider::action_type;

    std::unordered_multimap<path, path> tree;

    mock_fs_provider() = default;

    template <typename... Tail>
    mock_fs_provider(const char* name, Tail... tail) {
        add_path(name, tail...);
    }

    void add_path() const {}

    template <typename... Tail>
    void add_path(const char* name, Tail... tail) {
        if (!*name) return;

        const char* p = *name=='/'? name+1: name;

        for (const char* c = p; *p; p = c++) {
            while (*c && *c!='/') ++c;

            std::pair<path, path> entry{path{name, p}, path{name, c}};
            if (tree.find(entry.second)==tree.end()) {
                tree.insert(entry);
                tree.insert({entry.second, path{}});
            }
        }

        add_path(tail...);
    }

    static path canonical_key(const path& p) {
        return p.has_filename()? p: p.parent_path();
    }

    bool is_directory(const path& p) const {
        auto r = tree.equal_range(canonical_key(p));
        return r.first!=r.second && std::next(r.first)!=r.second;
    }

    bool exists(const path& p) const {
        return tree.find(canonical_key(p))!=tree.end();
    }

    void for_each_directory(const path& p, action_type action) const {
        auto r = tree.equal_range(canonical_key(p));
        for (auto i = r.first; i!=r.second; ++i) {
            auto entry = i->second;
            if (entry.empty()) continue;

            auto s = tree.equal_range(entry);
            if (s.first!=s.second && std::next(s.first)!=s.second) action(entry);
        }
    }

    void for_each_entry(const path& p, action_type action) const {
        auto r = tree.equal_range(canonical_key(p));
        for (auto i = r.first; i!=r.second; ++i) {
            auto entry = i->second;
            if (!entry.empty()) action(entry);
        }
    }
};

std::vector<path> sort_glob(const char* pattern, const glob_fs_provider& fs) {
    auto results = glob_basic(pattern, fs);
    std::sort(results.begin(), results.end());
    return results;
}

TEST(glob, simple_patterns) {
    glob_fs_provider fs = mock_fs_provider{"fish", "fop", "barf", "barry", "tip"};

    using pvector = std::vector<path>;

    EXPECT_EQ(pvector({"fish", "fop"}), sort_glob("f*", fs));
    EXPECT_EQ(pvector({"fop", "tip"}), sort_glob("??p", fs));
    EXPECT_EQ(pvector(), sort_glob("x*", fs));
}

TEST(glob, literals) {
    glob_fs_provider fs = mock_fs_provider{
        "/abc/def/ghi",
        "/abc/de",
        "/xyz",
        "pqrs/tuv/w",
        "pqrs/tuv/wxy"
    };

    using pvector = std::vector<path>;

    EXPECT_EQ(pvector({"/abc/def/ghi"}), sort_glob("/abc/def/ghi", fs));
    EXPECT_EQ(pvector({"/abc/def/ghi"}), sort_glob("/*/def/ghi", fs));
    EXPECT_EQ(pvector({"/abc/def/ghi"}), sort_glob("/*/*/ghi", fs));
    EXPECT_EQ(pvector({"/abc/def/ghi"}), sort_glob("/abc/def/*", fs));
    EXPECT_EQ(pvector({"/abc/def/ghi"}), sort_glob("/abc/*/*", fs));
    EXPECT_EQ(pvector({"pqrs/tuv/w", "pqrs/tuv/wxy"}), sort_glob("pqrs/tuv/w*", fs));
    EXPECT_EQ(pvector({"pqrs/tuv/w", "pqrs/tuv/wxy"}), sort_glob("*/tuv/w*", fs));
    EXPECT_EQ(pvector({"pqrs/tuv/w", "pqrs/tuv/wxy"}), sort_glob("pqrs/t*/w*", fs));
}

TEST(glob, multidir) {
    glob_fs_provider fs = mock_fs_provider{
        "abc/fab/x",
        "abc/fab/yz",
        "abc/flib/x",
        "abc/flib/yz",
        "abc/rib/x",
        "def/rib/yz",
        "def/fab/x",
        "def/fab/yz",
        "def/rib/x",
        "def/rib/yz"
    };

    using pvector = std::vector<path>;

    EXPECT_EQ(pvector({"abc/fab/x", "abc/flib/x"}), sort_glob("*c/f*b/?", fs));
}

TEST(glob, dots) {
    glob_fs_provider fs = mock_fs_provider{
        "f.oo/b.ar", "f.oo/.bar",
        ".foo/b.ar", ".foo/.bar"
    };

    using pvector = std::vector<path>;

    EXPECT_EQ(pvector({"f.oo/b.ar"}), sort_glob("*/*", fs));
    EXPECT_EQ(pvector({".foo/b.ar"}), sort_glob(".*/*", fs));
    EXPECT_EQ(pvector({"f.oo/b.ar"}), sort_glob("f[.z]oo/*", fs));
    EXPECT_EQ(pvector({"f.oo/b.ar"}), sort_glob("f?oo/*", fs));
    EXPECT_EQ(pvector(), sort_glob("[.z]foo/*", fs));
    EXPECT_EQ(pvector(), sort_glob("?foo/*", fs));
}

