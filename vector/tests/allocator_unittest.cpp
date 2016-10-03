#include "gtest.h"

#include <Allocator.hpp>
#include <HostCoordinator.hpp>

// verify that metafunction for memory alignment works
template <size_t s>
struct packer {
    char vals[s];
};

TEST(Allocator, minimum_possible_alignment) {
    using namespace memory::impl;
    size_t tmp;
    tmp = minimum_possible_alignment< packer<1> >();
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<2> >();
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<3> >();
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<4> >();
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<5> >();
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<6> >();
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<7> >();
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<8> >();
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<9> >();
    EXPECT_EQ(tmp,16);
    tmp = minimum_possible_alignment< packer<10> >();
    EXPECT_EQ(tmp,16);
    tmp = minimum_possible_alignment< packer<11> >();
    EXPECT_EQ(tmp,16);
    tmp = minimum_possible_alignment< packer<15> >();
    EXPECT_EQ(tmp,16);
    tmp = minimum_possible_alignment< packer<16> >();
    EXPECT_EQ(tmp,16);
    tmp = minimum_possible_alignment< packer<17> >();
    EXPECT_EQ(tmp,32);
    tmp = minimum_possible_alignment< packer<29> >();
    EXPECT_EQ(tmp,32);
    tmp = minimum_possible_alignment< packer<30> >();
    EXPECT_EQ(tmp,32);
    tmp = minimum_possible_alignment< packer<31> >();
    EXPECT_EQ(tmp,32);
    tmp = minimum_possible_alignment< packer<32> >();
    EXPECT_EQ(tmp,32);
    tmp = minimum_possible_alignment< packer<33> >();
    EXPECT_EQ(tmp,64);
    tmp = minimum_possible_alignment< packer<127> >();
    EXPECT_EQ(tmp,128);
    tmp = minimum_possible_alignment< packer<128> >();
    EXPECT_EQ(tmp,128);
    tmp = minimum_possible_alignment< packer<129> >();
    EXPECT_EQ(tmp,256);
    tmp = minimum_possible_alignment< packer<255> >();
    EXPECT_EQ(tmp,256);
    tmp = minimum_possible_alignment< packer<256> >();
    EXPECT_EQ(tmp,256);
    tmp = minimum_possible_alignment< packer<257> >();
    EXPECT_EQ(tmp,512);
}

TEST(Allocator, padding) {
    using namespace memory::impl;

    EXPECT_EQ( get_padding<char>(8,  0), 0 );
    EXPECT_EQ( get_padding<char>(8,  1), 7 );
    EXPECT_EQ( get_padding<char>(8,  2), 6 );
    EXPECT_EQ( get_padding<char>(8,  3), 5 );
    EXPECT_EQ( get_padding<char>(8,  4), 4 );
    EXPECT_EQ( get_padding<char>(8,  5), 3 );
    EXPECT_EQ( get_padding<char>(8,  6), 2 );
    EXPECT_EQ( get_padding<char>(8,  7), 1 );
    EXPECT_EQ( get_padding<char>(8,  8), 0 );

    EXPECT_EQ( get_padding<char>(8,  9), 7 );
    EXPECT_EQ( get_padding<char>(8, 10), 6 );
    EXPECT_EQ( get_padding<char>(8, 11), 5 );
    EXPECT_EQ( get_padding<char>(8, 12), 4 );
    EXPECT_EQ( get_padding<char>(8, 13), 3 );
    EXPECT_EQ( get_padding<char>(8, 14), 2 );
    EXPECT_EQ( get_padding<char>(8, 15), 1 );
    EXPECT_EQ( get_padding<char>(8, 16), 0 );

    EXPECT_EQ( get_padding<double>(64, 0), 0 );
    EXPECT_EQ( get_padding<double>(64, 1), 7 );
    EXPECT_EQ( get_padding<double>(64, 2), 6 );
    EXPECT_EQ( get_padding<double>(64, 3), 5 );
    EXPECT_EQ( get_padding<double>(64, 4), 4 );
    EXPECT_EQ( get_padding<double>(64, 5), 3 );
    EXPECT_EQ( get_padding<double>(64, 6), 2 );
    EXPECT_EQ( get_padding<double>(64, 7), 1 );
    EXPECT_EQ( get_padding<double>(64, 8), 0 );
}
