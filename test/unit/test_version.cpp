#include <regex>
#include <string>

#include <gtest/gtest.h>

#include <arbor/version.hpp>

using namespace std::string_literals;
using std::regex;
using std::regex_search;
using std::string;
using std::to_string;

TEST(version, libmatch) {
    string header_version = ARB_VERSION;
    string header_source_id = ARB_SOURCE_ID;
    string header_arch = ARB_ARCH;
    string header_build_config = ARB_BUILD_CONFIG;
    string header_full_build_id = ARB_FULL_BUILD_ID;
#ifdef ARB_VERSION_DEV
    string header_version_dev = ARB_VERSION_DEV;
    EXPECT_FALSE(header_version_dev.empty());
#else
    string header_version_dev;
#endif
    int header_version_major = ARB_VERSION_MAJOR;
    int header_version_minor = ARB_VERSION_MINOR;
    int header_version_patch = ARB_VERSION_PATCH;

    string lib_version = arb::version;
    string lib_source_id = arb::source_id;
    string lib_arch = arb::arch;
    string lib_build_config = arb::build_config;
    string lib_full_build_id = arb::full_build_id;
    constexpr int lib_version_major = arb::version_major;
    constexpr int lib_version_minor = arb::version_minor;
    constexpr int lib_version_patch = arb::version_patch;
    string lib_version_dev = arb::version_dev;

    EXPECT_EQ(header_version, lib_version);
    EXPECT_EQ(header_source_id, lib_source_id);
    EXPECT_EQ(header_arch, lib_arch);
    EXPECT_EQ(header_build_config, lib_build_config);
    EXPECT_EQ(header_full_build_id, lib_full_build_id);
    EXPECT_EQ(header_version_major, lib_version_major);
    EXPECT_EQ(header_version_minor, lib_version_minor);
    EXPECT_EQ(header_version_patch, lib_version_patch);
    EXPECT_EQ(header_version_dev, lib_version_dev);
}

TEST(version, sane_config) {
    EXPECT_TRUE(arb::build_config=="DEBUG"s || arb::build_config=="RELEASE"s);
}

TEST(version, version_components) {
    string dev = arb::version_dev;

    if (arb::version_patch>0) {
        auto expected = to_string(arb::version_major)+"."+to_string(arb::version_minor)+"."+to_string(arb::version_patch);
        expected += dev.empty()? "": "-"+dev;

        EXPECT_EQ(expected, ARB_VERSION);
    }
    else {
        auto expected_majmin = to_string(arb::version_major)+"."+to_string(arb::version_minor);
        auto expected_suffix = dev.empty()? "": "-"+dev;

        EXPECT_TRUE(expected_majmin+expected_suffix==ARB_VERSION || expected_majmin+".0"+expected_suffix==ARB_VERSION);
    }
}

TEST(version, full_build_id) {
    EXPECT_TRUE(regex_search(arb::full_build_id, regex("(;|^)config=.")));
    EXPECT_TRUE(regex_search(arb::full_build_id, regex("(;|^)version=.")));
    EXPECT_TRUE(regex_search(arb::full_build_id, regex("(;|^)source_id=.")));
    EXPECT_TRUE(regex_search(arb::full_build_id, regex("(;|^)arch=.")));
}

