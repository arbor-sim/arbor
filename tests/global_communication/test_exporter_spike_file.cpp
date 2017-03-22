#include "../gtest.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <io/exporter_spike_file.hpp>
#include <spike.hpp>

class exporter_spike_file_fixture : public ::testing::Test {
protected:
    using communicator_type = nest::mc::communication::global_policy;

    using exporter_type =
        nest::mc::io::exporter_spike_file<communicator_type>;

    std::string file_name_;
    std::string path_;
    std::string extension_;
    unsigned index_;

    exporter_spike_file_fixture() :
        file_name_("spikes_exporter_spike_file_fixture"),
        path_("./"),
        extension_("gdf"),
        index_(communicator_type::id())
    {}

    std::string get_standard_file_name() {
        return exporter_type::create_output_file_path(file_name_, path_, extension_, index_);
    }

    void SetUp() {
        // code here will execute just before the test ensues 
    }

    void TearDown() {
        // delete the start create file
        std::remove(get_standard_file_name().c_str());
    }

    ~exporter_spike_file_fixture()
    {}
};

TEST_F(exporter_spike_file_fixture, constructor) {
    exporter_type exporter(file_name_, path_, extension_, true);

    //test if the file exist and depending on over_write throw or delete
    std::ifstream f(get_standard_file_name());
    EXPECT_TRUE(f.good());

    // We now know the file exists, so create a new exporter with overwrite false
    try {
        exporter_type exporter1(file_name_, path_, extension_, false);
        FAIL() << "expected a file already exists error";
    }
    catch (const std::runtime_error& err) {
        EXPECT_EQ(
            err.what(),
            "Tried opening file for writing but it exists and over_write is false: " +
            get_standard_file_name()
        );
    }
    catch (...) {
        FAIL() << "expected a file already exists error";
    }
}

TEST_F(exporter_spike_file_fixture, create_output_file_path) {
    // Create some random paths, no need for fancy tests here
    std::string produced_filename =
        exporter_type::create_output_file_path("spikes", "./", "gdf", 0);
    EXPECT_STREQ(produced_filename.c_str(), "./spikes_0.gdf");

    produced_filename =
        exporter_type::create_output_file_path("a_name", "../../", "txt", 5);
    EXPECT_STREQ(produced_filename.c_str(), "../../a_name_5.txt");
}

TEST_F(exporter_spike_file_fixture, do_export) {
    {
        exporter_type exporter(file_name_, path_, extension_);

        // Create some spikes
        std::vector<nest::mc::spike> spikes;
        spikes.push_back({ { 0, 0 }, 0.0 });
        spikes.push_back({ { 0, 0 }, 0.1 });
        spikes.push_back({ { 1, 0 }, 1.0 });
        spikes.push_back({ { 1, 0 }, 1.1 });

        // now do the export
        exporter.output(spikes);
    }

    // Test if we have spikes in the file?
    std::ifstream f(get_standard_file_name());
    EXPECT_TRUE(f.good());

    std::string line;

    EXPECT_TRUE(std::getline(f, line));
    EXPECT_STREQ(line.c_str(), "0 0.0000");
    EXPECT_TRUE(std::getline(f, line));
    EXPECT_STREQ(line.c_str(), "0 0.1000");
    EXPECT_TRUE(std::getline(f, line));
    EXPECT_STREQ(line.c_str(), "1 1.0000");
    EXPECT_TRUE(std::getline(f, line));
    EXPECT_STREQ(line.c_str(), "1 1.1000");
}
