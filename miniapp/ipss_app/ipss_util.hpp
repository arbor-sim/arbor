#include <fstream>
#include <vector>
#include <utility>
#include <common_types.hpp>
#include <exception>
#include <iostream>

namespace arb {
    class csv_error : public std::exception
    {
    public:
        csv_error(char *what) : what_(what) {}

    private:
        virtual const char* what() const throw()
        {
            return what_ ;
        }

    private:
        char* what_;
    };

    std::vector<std::pair<time_type, double>> parse_time_pair_in_path(std::string path) {
        std::ifstream infile(path);
        arb::time_type time;
        double rate;
        char comma;

        std::vector<std::pair<time_type, double>> pairs;

        if (infile) {
            while (infile >> time >> comma >> rate) {
                pairs.push_back({ time,rate });
            }
        }
        else {
            throw csv_error("Could not open supplied csv file");
        }

        return pairs;
    }

}


