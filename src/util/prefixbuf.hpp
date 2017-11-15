#pragma once

// Output-only stream buffer that prepends a prefix to each line of output,
// together with stream manipulators for setting the prefix and managing
// an indentation level.

#include <ostream>
#include <sstream>
#include <string>

namespace arb {
namespace util {

class prefixbuf: public std::streambuf {
public:
    explicit prefixbuf(std::streambuf* inner): inner_(inner) {}

    prefixbuf(prefixbuf&&) = default;
    prefixbuf(const prefixbuf&) = delete;

    prefixbuf& operator=(prefixbuf&&) = default;
    prefixbuf& operator=(const prefixbuf&) = delete;

    std::streambuf* inner() { return inner_; }
    std::string prefix;

protected:
    std::streambuf* inner_;
    bool bol_ = true;

    std::streamsize xsputn(const char_type* s, std::streamsize count) override;
    int_type overflow(int_type ch) override;
};

// Manipulators:
//
// setprefix(s):   explicitly set prefix string on corresponding prefixbuf.
// indent:         increase indentation level by one tab width.
// indent(n):      increase indentation level by n tab widths.
// popindent:      undo last `indent` operation.
// popindent(n)    undo last n `indent` operations.
// settab(w):      set tab width to w (default is 4).
//
// Note that the prefix string is a property of the prefixbuf, not the stream,
// and so will not be preserved by e.g. `copyfmt`.

class setprefix {
public:
    explicit setprefix(std::string prefix): prefix_(std::move(prefix)) {}

    friend std::ostream& operator<<(std::ostream& os, const setprefix& sp);

private:
    std::string prefix_;
};

struct indent_manip {
    enum action_enum {push, pop, settab};

    explicit constexpr indent_manip(action_enum action, unsigned value=0):
        action_(action), value_(value)
    {}

    // convenience interface: allows using both `indent` and `indent(n)`
    // as stream manipulators.
    indent_manip operator()(unsigned n) const {
        return indent_manip(action_, n);
    }

    friend std::ostream& operator<<(std::ostream& os, indent_manip in);

private:
    static constexpr unsigned default_tabwidth = 4;
    static int xindex();

    action_enum action_;
    unsigned value_;
};

constexpr indent_manip indent{indent_manip::push, 1u};
constexpr indent_manip popindent{indent_manip::pop, 1u};

inline indent_manip settab(unsigned w) {
    return indent_manip{indent_manip::settab, w};
}

// Wrap an ostringstream with a prefixbuf.

class pfxstringstream: public std::ostream {
public:
    pfxstringstream():
        std::ostream(&pbuf_),
        sbuf_(std::ios_base::out),
        pbuf_(&sbuf_)
    {}

    pfxstringstream(pfxstringstream&&) = default;
    pfxstringstream& operator=(pfxstringstream&&) = default;

    std::string str() const { return sbuf_.str(); }
    void str(const std::string& s) { sbuf_.str(s); }

    prefixbuf* rdbuf() { return &pbuf_; }

private:
    std::stringbuf sbuf_;
    prefixbuf pbuf_;
};


} // namespace util
} // namespace arb
