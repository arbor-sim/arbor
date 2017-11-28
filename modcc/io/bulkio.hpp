#pragma once

// Read or write the contents of a file in toto.

#include <string>
#include <iterator>
#include <fstream>

namespace io {

template <typename HasAssign>
void snarf(std::istream& in, HasAssign& A) {
    A.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

template <typename HasAssign>
void snarf(const std::string& filename, HasAssign& A) {
    std::ifstream fs;
    fs.exceptions(std::ios::failbit);
    fs.open(filename);
    snarf(fs, A);
}

inline std::string snarf(std::istream& in) {
    std::string s;
    snarf(in, s);
    return s;
}

inline std::string snarf(const std::string& filename) {
    std::string s;
    snarf(filename, s);
    return s;
}

template <typename Container>
void blat(const Container& data, std::ostream& out) {
    std::copy(std::begin(data), std::end(data), std::ostreambuf_iterator<char>(out));
}

template <typename Container>
void blat(const Container& data, const std::string& filename) {
    std::ofstream fs;
    fs.exceptions(std::ios::failbit);
    fs.open(filename);
    blat(data, fs);
}

}
