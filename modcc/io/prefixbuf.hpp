#pragma once

// Output-only stream buffer that prepends a prefix to each line of output,
// together with stream manipulators for setting the prefix and managing
// an indentation level.

#include <ostream>
#include <sstream>
#include <string>

#include <libmodcc/export.hpp>

namespace io {

// `prefixbuf` acts an output-only filter for another streambuf, inserting
// the contents of the `prefix` string before the first character in a line.
//
// The following code, for example:
//
//     prefixbuf p(std::cout.rdbuf());
//     std::cout.rdbuf(&p);
//     p.prefix = ">>> ";
//     std::cout << "hello\nworld\n";
//
// would emit to stdout:
//
//     >>> hello
//     >>> world
//
// A flag determines if the prefixbuf should or should not emit the prefix
// for empty lines.

class ARB_LIBMODCC_API prefixbuf: public std::streambuf {
public:
    explicit prefixbuf(std::streambuf* inner, bool prefix_empty_lines=false):
        inner_(inner), prefix_empty_lines_(prefix_empty_lines) {}

    prefixbuf(prefixbuf&& other): std::streambuf(other) {
        prefix = std::move(other.prefix);
        inner_ = other.inner_;
        prefix_empty_lines_ = other.prefix_empty_lines_;
        bol_ = other.bol_;
    }

    prefixbuf(const prefixbuf&) = delete;

    prefixbuf& operator=(prefixbuf&&) = delete;
    prefixbuf& operator=(const prefixbuf&) = delete;

    std::streambuf* inner() { return inner_; }
    std::string prefix;

protected:
    std::streambuf* inner_;
    bool prefix_empty_lines_ = false;
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
//
// All but `setprefix` are implemented as values of type `indent_manip`
// below.
//
// The manipulator `indent(0)` can be used to reset the prefix of the underlying
// stream to match the current indentation level.

class ARB_LIBMODCC_API setprefix {
public:
    explicit setprefix(std::string prefix): prefix_(std::move(prefix)) {}

    friend std::ostream& operator<<(std::ostream& os, const setprefix& sp);

private:
    std::string prefix_;
};

struct ARB_LIBMODCC_API indent_manip {
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

// Wrap a stringbuf with a prefixbuf, and present as a stream.
// Acts very much like a `std::ostringstream`, but with prefix
// and indent functionality.

class ARB_LIBMODCC_API pfxstringstream: public std::ostream {
public:
    pfxstringstream():
        std::ostream(nullptr),
        sbuf_(std::ios_base::out),
        pbuf_(&sbuf_)
    {
        std::ostream::rdbuf(&pbuf_);
    }

    pfxstringstream(pfxstringstream&& other):
        std::ostream(std::move(other)),
        sbuf_(std::move(other.sbuf_)),
        pbuf_(std::move(other.pbuf_))
    {
        std::ostream::rdbuf(&pbuf_);
    }

    std::string str() const { return sbuf_.str(); }
    void str(const std::string& s) { sbuf_.str(s); }

    prefixbuf* rdbuf() { return &pbuf_; }

private:
    std::stringbuf sbuf_;
    prefixbuf pbuf_;
};

} // namespace io
