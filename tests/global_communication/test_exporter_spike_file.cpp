#include "gtest.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <communication/exporter_spike_file.hpp>

class exporter_spike_file_fixture : public ::testing::Test {

protected:
    using time_type = float;
    using communicator_type = nest::mc::communication::global_policy;

    using spike_type = nest::mc::communication::exporter_spike_file<time_type, 
        communicator_type>::spike_type;

    using exporter_type = nest::mc::communication::exporter_spike_file<time_type, communicator_type>;
    std::string file_name;
    std::string path;
    std::string extention;
    unsigned index;

    exporter_spike_file_fixture()
        :
        file_name("spikes_exporter_spike_file_fixture"),
        path("./"),
        extention("gdf"),
        index(0)
    {}

    std::string get_standard_file_name()
    {
        return exporter_type::create_output_file_path(
            file_name, path, extention, 0);
    }

    void SetUp() {
        // code here will execute just before the test ensues 
    }

    void TearDown() {
        // delete the start create file
        std::remove(get_standard_file_name().c_str());
    }

    ~exporter_spike_file_fixture() {

    }
};

TEST_F(exporter_spike_file_fixture, constructor)
{
    
    exporter_type exporter(file_name, path, extention, true);

    // after construction the state of the exporter should be valid
    EXPECT_TRUE(exporter.ok());


    //test if the file exist and depending on over_write throw or delete
    std::ifstream f(get_standard_file_name());

    
    EXPECT_TRUE(f.good());
}

TEST_F(exporter_spike_file_fixture, create_output_file_path)
{
    // Create some random paths, no need for fancy tests here
    std::string produced_filename =
        exporter_type::create_output_file_path(
            "spikes", "./", "gdf", 0);
    EXPECT_STREQ(produced_filename.c_str(), "./spikes_0.gdf");

    produced_filename =
        exporter_type::create_output_file_path(
            "a_name", "../../", "txt", 5);
    EXPECT_STREQ(produced_filename.c_str(), "../../a_name_5.txt");
}

TEST_F(exporter_spike_file_fixture, do_export)
{
    
    
    exporter_type exporter(file_name,
        path, extention);

    // Create some spikes
    std::vector<spike_type> spikes;
    spikes.push_back({ { 0, 0 }, 0.0});
    spikes.push_back({ { 0, 0 }, 0.1 });
    spikes.push_back({ { 1, 0 }, 1.0 });
    spikes.push_back({ { 1, 0 }, 1.1 });

    // add to the exporter
    exporter.add_data(spikes);

    // now do the export
    exporter.do_export();
    
    // Test if we have spikes in the file
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
