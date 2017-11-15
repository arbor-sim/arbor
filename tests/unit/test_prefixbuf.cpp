#include "../gtest.h"

#include <cstring>
#include <iomanip>
#include <string>
#include <sstream>

#include <util/prefixbuf.hpp>

using namespace arb::util;

TEST(prefixbuf, prefix) {
    auto write_sputn = [](std::streambuf& b, const char* c) {
        b.sputn(c, std::strlen(c));
    };

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

TEST(prefixbuf, pfxstringstream_str) {
    pfxstringstream p;
    p.str("0123456789");
    p.rdbuf()->prefix = "__";
    p << "a\nb";

    std::string expected = "__a\n__b789";
    EXPECT_EQ(expected, p.str());
}

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
