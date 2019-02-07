#include "../gtest.h"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <sup/path.hpp>

using namespace sup;

TEST(path, posix_ctor) {
    // test constructor ans assignment overloads with sample character sequences.
    posix_path p1;
    EXPECT_EQ("", p1.native());
    EXPECT_TRUE(p1.empty());

    const char* cs = "foo/bar";
    std::string str_cs(cs);
    std::vector<char> vec_cs(str_cs.begin(), str_cs.end());

    posix_path p2(cs);
    posix_path p3(str_cs);
    posix_path p4(str_cs.begin(), str_cs.end());
    posix_path p5(vec_cs.begin(), vec_cs.end());

    EXPECT_FALSE(p2.empty());
    EXPECT_EQ(str_cs, p2.native());
    EXPECT_EQ(str_cs, p3.native());
    EXPECT_EQ(str_cs, p4.native());
    EXPECT_EQ(str_cs, p5.native());

    posix_path p6(p2);
    EXPECT_EQ(str_cs, p6.native());

    posix_path p7(std::move(p6));
    EXPECT_EQ(str_cs, p7.native());

    // test operator= overloads (and ref return values)
    posix_path p;
    EXPECT_EQ(str_cs, (p=p2).native());
    EXPECT_EQ(str_cs, (p=cs).native());
    EXPECT_EQ(str_cs, (p=str_cs).native());
    EXPECT_EQ(str_cs, (p=std::move(p7)).native());

    // test assign overloads (and ref return values)
    EXPECT_EQ(str_cs, p.assign(p2).native());
    EXPECT_EQ(str_cs, p.assign(cs).native());
    EXPECT_EQ(str_cs, p.assign(str_cs).native());
    EXPECT_EQ(str_cs, p.assign(vec_cs.begin(), vec_cs.end()).native());
}

TEST(path, posix_native) {
    // native path should match string argument exactly
    std::string ps = "/abs/path";
    std::string qs = "rel/path.ext";

    EXPECT_EQ(ps, posix_path{ps}.native());
    EXPECT_EQ(qs, posix_path{qs}.native());

    // default string conversion
    std::string ps_bis = posix_path{ps};
    std::string qs_bis = posix_path{qs};

    EXPECT_EQ(ps, ps_bis);
    EXPECT_EQ(qs, qs_bis);

    // cstr
    posix_path ps_path{ps};
    EXPECT_TRUE(!std::strcmp(ps_path.c_str(), ps.c_str()));
}

TEST(path, posix_generic) {
    // expect native and generic to be same for POSIX paths
    path p("/abs/path"), q("rel/path.ext");

    EXPECT_EQ(p.generic_string(), p.native());
    EXPECT_EQ(q.generic_string(), q.native());
}

TEST(path, posix_append) {
    posix_path p1{""}, q1{"rel"};

    posix_path p(p1);
    p.append(q1);
    EXPECT_EQ(p1/q1, p);

    p = p1;
    p /= q1;
    EXPECT_EQ(p1/q1, p);
    EXPECT_EQ("rel", p.native());

    posix_path p2{"ab"}, q2{"rel"};

    p = p2;
    p.append(q2);
    EXPECT_EQ(p2/q2, p);

    p = p2;
    p /= q2;
    EXPECT_EQ(p2/q2, p);
    EXPECT_EQ("ab/rel", p.native());

    EXPECT_EQ("foo/bar", (posix_path("foo/")/posix_path("/bar")).native());
    EXPECT_EQ("foo/bar", (posix_path("foo")/posix_path("/bar")).native());
    EXPECT_EQ("foo/bar", (posix_path("foo/")/posix_path("bar")).native());
    EXPECT_EQ("/foo/bar/", (posix_path("/foo/")/posix_path("/bar/")).native());
}

TEST(path, compare) {
    posix_path p1("/a/b"), p2("/a//b"), p3("/a/b/c/."), p4("/a/b/c/"), p5("a/bb/c"), p6("a/b/c/");

    EXPECT_EQ(p1, p2);
    EXPECT_LE(p1, p2);
    EXPECT_GE(p1, p2);
    EXPECT_EQ(p3, p4);
    EXPECT_LE(p3, p4);
    EXPECT_GE(p3, p4);

    EXPECT_LT(p1, p3);
    EXPECT_LE(p1, p3);
    EXPECT_GT(p3, p1);
    EXPECT_GE(p3, p1);
    EXPECT_NE(p3, p1);

    EXPECT_NE(p4, p6);

    EXPECT_LT(p4, p5);
    EXPECT_LE(p4, p5);
    EXPECT_GT(p5, p4);
    EXPECT_GE(p5, p4);
    EXPECT_NE(p5, p4);
}

TEST(path, posix_concat) {
    posix_path p1{""}, q1{"tail"};

    posix_path p(p1);
    p.concat(q1);
    EXPECT_EQ("tail", p.native());

    p = p1;
    p += q1;
    EXPECT_EQ("tail", p.native());

    posix_path p2{"ab"}, q2{"cd"};

    p = p2;
    p.concat(q2);
    EXPECT_EQ("abcd", p.native());

    p = p2;
    p += q2;
    EXPECT_EQ("abcd", p.native());
}

TEST(path, posix_absrel_query) {
    posix_path p1("/abc/def");
    EXPECT_FALSE(p1.is_relative());
    EXPECT_TRUE(p1.is_absolute());

    posix_path p2("abc/def");
    EXPECT_TRUE(p2.is_relative());
    EXPECT_FALSE(p2.is_absolute());

    posix_path p3("");
    EXPECT_TRUE(p3.is_relative());
    EXPECT_FALSE(p3.is_absolute());

    posix_path p4("/");
    EXPECT_FALSE(p4.is_relative());
    EXPECT_TRUE(p4.is_absolute());

    posix_path p5("..");
    EXPECT_TRUE(p3.is_relative());
    EXPECT_FALSE(p3.is_absolute());
}

TEST(path, posix_swap) {
    posix_path p1("foo"), p2("/bar");
    p1.swap(p2);

    EXPECT_EQ("foo", p2.native());
    EXPECT_EQ("/bar", p1.native());

    swap(p1, p2);

    EXPECT_EQ("foo", p1.native());
    EXPECT_EQ("/bar", p2.native());
}

TEST(path, filename) {
    auto filename = [](auto p) { return posix_path(p).filename().native(); };
    auto has_filename = [](auto p) { return posix_path(p).has_filename(); };

    EXPECT_EQ("foo", filename("foo"));
    EXPECT_TRUE(has_filename("foo"));

    EXPECT_EQ("foo", filename("bar/foo"));
    EXPECT_TRUE(has_filename("bar/foo"));

    EXPECT_EQ("foo", filename("/bar/foo"));
    EXPECT_TRUE(has_filename("/bar/foo"));

    EXPECT_EQ("foo", filename("./foo"));
    EXPECT_TRUE(has_filename("./foo"));

    EXPECT_EQ("foo", filename("../foo"));
    EXPECT_TRUE(has_filename("../foo"));

    EXPECT_EQ(".", filename("."));
    EXPECT_TRUE(has_filename("."));

    EXPECT_EQ("", filename("foo/"));
    EXPECT_FALSE(has_filename("foo/"));

    EXPECT_EQ("", filename("foo/bar/"));
    EXPECT_FALSE(has_filename("foo/bar/"));

    EXPECT_EQ("", filename("/foo/bar/"));
    EXPECT_FALSE(has_filename("/foo/bar/"));

    EXPECT_EQ("", filename("./"));
    EXPECT_FALSE(has_filename("./"));

    EXPECT_EQ("", filename("/"));
    EXPECT_FALSE(has_filename("/"));
}

TEST(path, parent_path) {
    auto parent_path = [](auto p) { return posix_path(p).parent_path().native(); };
    auto has_parent_path = [](auto p) { return posix_path(p).has_parent_path(); };

    EXPECT_EQ("/abc", parent_path("/abc/"));
    EXPECT_TRUE(has_parent_path("/abc/"));

    EXPECT_EQ("/abc", parent_path("/abc/def"));
    EXPECT_TRUE(has_parent_path("/abc/def"));

    EXPECT_EQ("/abc", parent_path("/abc/."));
    EXPECT_TRUE(has_parent_path("/abc/."));

    EXPECT_EQ("/", parent_path("/"));
    EXPECT_TRUE(has_parent_path("/"));

    EXPECT_EQ("abc", parent_path("abc/def"));
    EXPECT_TRUE(has_parent_path("abc/def"));

    EXPECT_EQ("abc/def", parent_path("abc/def/ghi"));
    EXPECT_TRUE(has_parent_path("abc/def/ghi"));

    EXPECT_EQ("", parent_path("abc"));
    EXPECT_FALSE(has_parent_path("abc"));

    EXPECT_EQ("", parent_path("."));
    EXPECT_FALSE(has_parent_path("."));

    EXPECT_EQ("", parent_path(""));
    EXPECT_FALSE(has_parent_path(""));
}

TEST(path, posix_iostream) {
    std::istringstream ss("/quux/xyzzy");
    posix_path p;
    ss >> p;
    EXPECT_EQ("/quux/xyzzy", p.native());

    std::ostringstream uu;
    uu << p;
    EXPECT_EQ("/quux/xyzzy", uu.str());
}

TEST(path, filesystem_error) {
    auto io_error = std::make_error_code(std::errc::io_error);

    filesystem_error err0("err0", io_error);
    filesystem_error err1("err1", "/one", io_error);
    filesystem_error err2("err2", "/one", "/two", io_error);

    EXPECT_TRUE(dynamic_cast<std::system_error*>(&err0));
    EXPECT_NE(std::string::npos, std::string(err0.what()).find("err0"));
    EXPECT_EQ(path{}, err0.path1());
    EXPECT_EQ(path{}, err0.path2());
    EXPECT_EQ(io_error, err0.code());

    EXPECT_NE(std::string::npos, std::string(err1.what()).find("err1"));
    EXPECT_EQ(path("/one"), err1.path1());
    EXPECT_EQ(path{}, err1.path2());
    EXPECT_EQ(io_error, err1.code());

    EXPECT_NE(std::string::npos, std::string(err2.what()).find("err2"));
    EXPECT_EQ(path("/one"), err2.path1());
    EXPECT_EQ(path("/two"), err2.path2());
    EXPECT_EQ(io_error, err2.code());
}

TEST(path, posix_status) {
    // Expect (POSIX) filesystem to have:
    //      /            directory
    //      .            directory
    //      /dev/null    character special file

    try {
        file_status root = status("/");
        file_status dot = status(".");
        file_status dev_null = status("/dev/null");
        // file and /none/ should not exist, but we don't expect an error
        file_status nonesuch = status("/none/such");

        EXPECT_EQ(file_type::directory, root.type());
        EXPECT_EQ(file_type::directory, dot.type());
        EXPECT_EQ(file_type::character, dev_null.type());
        EXPECT_EQ(file_type::not_found, nonesuch.type());
    }
    catch (filesystem_error& e) {
        FAIL() << "unexpected error with status(): " << e.what();
    }

    try {
        file_status empty = status("");
        (void)empty;
        // should throw error wrapping ENOENT
        FAIL() << "status() did not throw on empty string";
    }
    catch (filesystem_error& e) {
        EXPECT_TRUE(e.code().default_error_condition()==std::errc::no_such_file_or_directory);
    }

    try {
        std::string way_too_long(1<<20,'z');
        file_status oops = status(way_too_long);
        (void)oops;
        // should throw error wrapping ENAMETOOLONG
        FAIL() << "status() did not throw on stupendously long file name";
    }
    catch (filesystem_error& e) {
        EXPECT_EQ(e.code().default_error_condition(), std::errc::filename_too_long);
    }
    try {
        std::error_code ec;
        file_status empty = status("", ec);
        EXPECT_EQ(file_type::none, empty.type());
        EXPECT_EQ(ec.default_error_condition(), std::errc::no_such_file_or_directory);
    }
    catch (filesystem_error& e) {
        FAIL() << "status(path, ec) should not throw";
    }
}

TEST(path, is_wrappers) {
    EXPECT_TRUE(exists("/"));
    EXPECT_TRUE(is_directory("/"));
    EXPECT_FALSE(is_regular_file("/"));

    EXPECT_TRUE(exists("/dev/null"));
    EXPECT_TRUE(is_character_file("/dev/null"));
    EXPECT_FALSE(is_regular_file("/"));

    EXPECT_FALSE(exists("/none/such"));
    EXPECT_FALSE(is_regular_file("/"));
    EXPECT_TRUE(is_directory("/"));

    EXPECT_THROW(exists(""), filesystem_error);
}

TEST(path, permissions) {
    perms p = perms::owner_read | perms::owner_write | perms::owner_exec;
    EXPECT_EQ(perms::owner_all, p);
    EXPECT_EQ(perms::none, p & perms::group_all);
    EXPECT_EQ(perms::none, p & perms::others_all);

    p = perms::group_read | perms::group_write | perms::group_exec;
    EXPECT_EQ(perms::group_all, p);
    EXPECT_EQ(perms::none, p & perms::owner_all);
    EXPECT_EQ(perms::none, p & perms::others_all);

    p = perms::others_read | perms::others_write | perms::others_exec;
    EXPECT_EQ(perms::others_all, p);
    EXPECT_EQ(perms::none, p & perms::owner_all);
    EXPECT_EQ(perms::none, p & perms::group_all);
}

TEST(path, posix_status_perms) {
    // Expect /dev/null permissions to be 0666
    perms null_perm = status("/dev/null").permissions();
    perms expected = perms::owner_read|perms::owner_write|perms::group_read|perms::group_write|perms::others_read|perms::others_write;
    EXPECT_EQ(expected, null_perm);

    // Expect / to be have exec flag set for everyone
    perms root_perm = status("/").permissions();
    EXPECT_NE(perms::none, root_perm&perms::owner_exec);
    EXPECT_NE(perms::none, root_perm&perms::group_exec);
    EXPECT_NE(perms::none, root_perm&perms::others_exec);
}

TEST(path, posix_directory_iterators) {
    // Expect that /dev exists and that iterating on /dev will give
    // an entry called 'null'. (This is guaranteed by POSIX.)
    //
    // Expect that the file type checks as given by is_block_file(),
    // is_fifo() etc. will agree for directory iterators and paths.

    auto it = directory_iterator("/dev");
    EXPECT_NE(directory_iterator(), it); // Equal => empty directory.

    bool found_dev_null = false;
    for (; it!=directory_iterator(); ++it) {
        if (it->path()=="/dev/null") found_dev_null = true;

        file_status st = symlink_status(it->path());

        // Check file type tests match up.
        EXPECT_EQ(it->is_block_file(), is_block_file(st));
        EXPECT_EQ(it->is_directory(), is_directory(st));
        EXPECT_EQ(it->is_character_file(), is_character_file(st));
        EXPECT_EQ(it->is_fifo(), is_fifo(st));
        EXPECT_EQ(it->is_regular_file(), is_regular_file(st));
        EXPECT_EQ(it->is_socket(), is_socket(st));
        EXPECT_EQ(it->is_symlink(), is_symlink(st));
    }

    EXPECT_TRUE(found_dev_null);
}
