#include <cstring>
#include <iomanip>
#include <string>
#include <sstream>

#include "io/prefixbuf.hpp"
#include "common.hpp"

using namespace io;

// Test public std::stringbuf 'put' interfaces on prefixbuf.

TEST(prefixbuf, prefix) {
    // Write a C-string with `std::steambuf::sputn` ---
    // exercises `prefixbuf::xsputn`.

    auto write_sputn = [](std::streambuf& b, const char* c) {
        b.sputn(c, std::strlen(c));
    };

    // Write a C-string with `std::steambuf::sputc` ---
    // exercises `prefixbuf::overflow`.

    auto write_sputc = [](std::streambuf& b, const char* c) {
        while (*c) {
            b.sputc(*c++);
        }
    };

    std::stringbuf s;
    write_sputn(s, "starting text\n");

    prefixbuf p(&s);
    p.prefix = ":) ";

    write_sputn(p, "foo\nbar ");
    write_sputc(p, "quux\nbaz ");
    p.prefix = "^^ ";
    write_sputn(p, "xyzzy\nplugh\n");

    std::string expected =
        "starting text\n"
        ":) foo\n"
        ":) bar quux\n"
        ":) baz xyzzy\n"
        "^^ plugh\n";

    EXPECT_EQ(expected, s.str());
}

// A prefixbuf can be configure to emit or not emit the
// prefix for empty lines.

TEST(prefixbuf, empty_lines) {
    auto write_sputn = [](std::streambuf& b, const char* c) {
        b.sputn(c, std::strlen(c));
    };

    std::stringbuf s;

    prefixbuf p1(&s, false); // omit prefix on blank lines
    p1.prefix = "1> ";
    write_sputn(p1, "hello\n\nfishies!\n\n");

    prefixbuf p2(&s, true); // include prefix on blank lines
    p2.prefix = "2> ";
    write_sputn(p2, "hello\n\nbunnies!\n");

    std::string expected =
        "1> hello\n"
        "\n"
        "1> fishies!\n"
        "\n"
        "2> hello\n"
        "2> \n"
        "2> bunnies!\n";

    EXPECT_EQ(expected, s.str());
}

// Test `pfxstringstream` basic functionality:
//
//   1. `rdbuf()` method gives pointer to `prefixbuf`.
//
//   2. Formatted write operations behave as expected,
//      but with the prefixbuf prefix-inserting behaviour.
//
//   3. `str()` method gives string from wrapped `std::stringbuf`.

TEST(prefixbuf, pfxstringstream) {
    pfxstringstream p;

    p.rdbuf()->prefix = "...";

    p << "_foo_ " << std::setw(5) << 123 << "\n";
    p << std::showbase << std::hex << 42;

    std::string expected =
        "..._foo_   123\n"
        "...0x2a";

    EXPECT_EQ(expected, p.str());
}

// Test that the `pfxstringstream::str(const std::string&)` method
// behaves analagously to that of `std::ostringstream`, viz. initializing
// the contents of the buffer.

TEST(prefixbuf, pfxstringstream_str) {
    pfxstringstream p;
    p.str("0123456789");
    p.rdbuf()->prefix = "__";
    p << "a\nb";

    std::string expected = "__a\n__b789";
    EXPECT_EQ(expected, p.str());
}

// Test indent manipulators against expected prefixes.

TEST(prefixbuf, indent_manip) {
    pfxstringstream p;

    p << settab(2);
    p << "0\n"
      << indent
      << "1\n"
      << indent(2)
      << "3\n"
      << indent
      << "4\n"
      << popindent(2)
      << "1\n"
      << popindent
      << "0";

    std::string expected =
        "0\n"
        "  1\n"
        "      3\n"
        "        4\n"
        "  1\n"
        "0";

    EXPECT_EQ(expected, p.str());
}

// `setprefix` goes behind the stream's back and sets the prefix
// in the underling streambuf directly.
//
// The stream's indentation will be re-applied to the streambuf
// when it encounters an indent manipulator; `indent(0)` would
// otherwise constitute a NOP.

TEST(prefixbuf, setprefix) {
    pfxstringstream p;

    p << indent << "one\ntwo ";
    p << setprefix("--->"); // override prefix
    p << "three\nfour ";
    p << indent(0)          // restore indentation
      << "five\nsix";

    std::string expected =
        "    one\n"
        "    two three\n"
        "--->four five\n"
        "    six";

    EXPECT_EQ(expected, p.str());
}

// Confirm the callback associated with the indentation state
// propagates the full indentation stack after `copyfmt`,
// and imposes the corresponding prefix on the underlying
// `prefixbuf`.

TEST(prefixbuf, copyfmt) {
    pfxstringstream p1;
    pfxstringstream p2;

    p1 << indent << "1\n" << indent << "2\n";
    p2 << "0\n";

    p2.copyfmt(p1);
    p2 << "2\n" << popindent << "1\n";

    p1 << indent << "3\n";

    std::string expected1 =
        "    1\n        2\n            3\n";

    std::string expected2 =
        "0\n        2\n    1\n";

    EXPECT_EQ(expected1, p1.str());
    EXPECT_EQ(expected2, p2.str());
}
