#include <string>

#include "../gtest.h"

#include <arbor/version.hpp>

TEST(version, libmatch) {
    using std::string;

    string header_version = ARB_VERSION;
    string header_source_id = ARB_SOURCE_ID;

    string lib_version = arb::version;
    string lib_source_id = arb::source_id;

    EXPECT_EQ(header_version, lib_version);
    EXPECT_EQ(header_source_id, lib_source_id);
}
