#include <list>
#include <string>
#include <vector>

#include <sup/glob.hpp>
#include <sup/path.hpp>

namespace sup {

struct glob_sup_fs_provider {
    using action_type = std::function<void (const sup::path&)>;

    bool is_directory(const sup::path& p) const {
        return sup::is_directory(p);
    };

    bool exists(const sup::path& p) const {
        return sup::exists(p);
    }

    void for_each_directory(const sup::path& p, action_type action) const {
        std::error_code ec;
        for (const auto& e: get_iterator(p)) {
            if (sup::is_directory(e.path(), ec)) action(e.path());
        }
    }

    void for_each_entry(const sup::path& p, action_type action) const {
        for (const auto& e: get_iterator(p)) {
            action(e.path());
        }
    }

private:
    static directory_iterator get_iterator(const sup::path& p) {
        return directory_iterator(p.empty()? ".": p,
            directory_options::skip_permission_denied);
    }
};

glob_fs_provider glob_native_provider{glob_sup_fs_provider{}};

static bool match_char_class(const char*& p, char c) {
    // Special cases for first character:
    // ! => negate test defined from following char.
    // - => treat '-' as literal.
    // ] => treat ']' as literal.

    if (*p!='[') return false;
    ++p;

    bool negate = false;
    bool match = false;

    if (*p=='!') {
        negate = true;
        ++p;
    }

    bool first = true;
    char lrange = 0;
    for (; !match && *p && (first || *p!=']'); ++p) {

        bool last = *p && p[1]==']';
        if (*p=='-' && lrange && !first && !last) {
            match = c>=lrange && c<=*++p;
            lrange = 0;
            continue;
        }

        lrange = *p;
        match = c==*p;
        first = false;
    }

    while (*p && *p!=']') ++p;
    if (!*p) return false;

    return match^negate;
}

// Special exception for filename globbing: an initial period '.' can only be matched
// by an intial '.' in the pattern.

bool glob_basic_match(const char* p, const char* t) {
     // NFA state represented by pointers into directly into pattern.
    std::list<const char*> state = {p};

    char c;
    bool initial_dot = *t=='.';
    do {
        c = *t++;
        for (auto i = state.begin(); i!=state.end();) {
            switch (**i) {
            case '*':
                if (initial_dot) goto fail;
                if (i==state.begin() || *std::prev(i)!=*i) {
                    state.insert(i, *i);
                }
                while (**i=='*') ++*i;
                continue;
            case '?':
                if (initial_dot) goto fail;
                if (c) goto advance;
                else goto fail;
            case '[':
                if (initial_dot) goto fail;
                if (c && match_char_class(*i, c)) goto advance;
                else goto fail;
            case '\\':
                ++*i; // fall-through
            default:
                if (**i==c) goto advance;
                else goto fail;
            }

        fail:
            i = state.erase(i);
            continue;

        advance:
            *i += !!c;
            ++i;
            continue;
        }
        initial_dot = false;
    } while (c && !state.empty());

    return !state.empty() && !*state.back();
}

// Return first component, overwriting delimitter with NUL.
// Set pattern to beginning of next path component, skipping delimiters.

struct pattern_component {
    const char* pattern = nullptr;
    bool literal = false;
    bool directory = false;
};

static pattern_component tokenize(char*& pattern) {
    if (!*pattern) return {pattern, true, false};

    char* s = nullptr;
    char* p = pattern;
    bool meta = false;

    do {
        while (*p=='/') ++p;

        bool in_charclass = false;
        bool escape = false;
        for (;*p && *p!='/'; ++p) {
            switch (*p) {
            case '[':
                if (!escape) {
                    in_charclass = true;
                    meta = true;
                }
                break;
            case '*':
                if (!escape) meta = true;
                break;
            case '?':
                if (!escape) meta = true;
                break;
            case '\\':
                if (!escape && !in_charclass) escape = true;
                break;
            case ']':
                if (in_charclass) in_charclass = false;
                break;
            default: ;
            }
        }
        if (!meta) s = p;
    } while (!meta && *p);

    pattern_component k = { pattern };
    k.literal = (bool)s;

    if (!s) s = p;
    k.directory = !!*s;

    pattern = s;
    while (*pattern=='/') ++pattern;

    *s = 0;
    return k;
}

// Return matching paths, unsorted, based on supplied pattern.
// Performs breadth-first search of the directory tree.

std::vector<path> glob_basic(const std::string& pattern, const glob_fs_provider& fs) {
    if (pattern.empty()) return {};

    // Make a mutable copy for tokenization.
    std::vector<char> pcopy(pattern.begin(), pattern.end());
    pcopy.push_back(0);

    char* c = pcopy.data();
    if (!*c) return {};

    std::vector<sup::path> paths, new_paths;
    paths.push_back("");

    if (*c=='/') {
        while (*c=='/') ++c;
        paths[0] = "/";
    }

    do {
        pattern_component component = tokenize(c);

        if (component.literal) {
            for (auto p: paths) {
                p /= component.pattern;

                if (component.directory) {
                    if (fs.is_directory(p)) new_paths.push_back(std::move(p));
                }
                else {
                    if (fs.exists(p)) new_paths.push_back(std::move(p));
                }
            }
        }
        else {
            auto push_if_match = [&new_paths, pattern = component.pattern](const sup::path& p) {
                if (glob_basic_match(pattern, p.filename().c_str())) new_paths.push_back(p);
            };

            for (auto p: paths) {
                if (component.directory) fs.for_each_directory(p.c_str(), push_if_match);
                else fs.for_each_entry(p.c_str(), push_if_match);
            }
        }

        std::swap(paths, new_paths);
        new_paths.clear();
    } while (*c);

    return paths;
}

} // namespace sup

