/**************************************************************
 * unit test driver
 **************************************************************/

#include <cstring>

#include "common.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (argc>1 && (!std::strcmp(argv[1],"-v") || !std::strcmp(argv[1],"--verbose"))) {
        g_verbose_flag = true;
    }
    return RUN_ALL_TESTS();
}

