#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "gtest.h"
#include "validation_data.hpp"

int usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " [-p|--path validation_data_directory]\n";
    return 1;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    if (argv[1] && (!strcmp(argv[1], "-p") || !strcmp(argv[1], "--path"))) {
        if (argv[2]) {
            testing::g_validation_data.set_path(argv[2]);
        }
        else {
            return usage(argv[0]);
        }
    }
    else if (argv[1]) {
        return usage(argv[0]);
    }

    return RUN_ALL_TESTS();
}
