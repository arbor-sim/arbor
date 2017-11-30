#pragma once

// Read or write the contents of a file in toto.

#include <string>
#include <iterator>
#include <fstream>

namespace io {

template <typename HasAssign>
void read_all(std::istream& in, HasAssign& A) {
    A.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

template <typename HasAssign>
void read_all(const std::string& filename, HasAssign& A) {
    std::ifstream fs;
    fs.exceptions(std::ios::failbit);
    fs.open(filename);
    read_all(fs, A);
}

inline std::string read_all(std::istream& in) {
    std::string s;
    read_all(in, s);
    return s;
}

inline std::string read_all(const std::string& filename) {
    std::string s;
    read_all(filename, s);
    return s;
}

template <typename Container>
void write_all(const Container& data, std::ostream& out) {
    std::copy(std::begin(data), std::end(data), std::ostreambuf_iterator<char>(out));
}

template <typename Container>
void write_all(const Container& data, const std::string& filename) {
    std::ofstream fs;
    fs.exceptions(std::ios::failbit);
    fs.open(filename);
    write_all(data, fs);
}

}
