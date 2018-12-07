#include "../gtest.h"

#include <sup/glob.hpp>
#include <sup/path.hpp>

using namespace sup;

#include <iterator>
#include <string>
#include <unordered_map>

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

