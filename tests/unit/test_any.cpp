#include "../gtest.h"

#include <iostream>

#include <util/any.hpp>

using namespace nest::mc;

TEST(any, copy_construction) {
    util::any any_int(2);
    EXPECT_EQ(any_int.type(), typeid(int));

    util::any any_float(2.0f);
    EXPECT_EQ(any_float.type(), typeid(float));

    std::string str = "hello";
    util::any any_string(str);
    EXPECT_EQ(any_string.type(), typeid(std::string));
}

namespace {

    struct moveable {
        moveable() = default;

        moveable(moveable&& other) {
            moves = other.moves+1;
            std::cout << "move " << moves << "\n";
        }

        moveable(const moveable& other) {
            copies = other.copies + 1;
            std::cout << "copy " << copies << "\n";
        }

        int moves=0;
        int copies=0;
    };

}

TEST(any, move_construction) {
    moveable m;

    util::any copied(m);
    util::any moved(std::move(m));

    util::any x(copied);
    util::any y(moved);
}
