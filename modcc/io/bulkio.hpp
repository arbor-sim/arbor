#pragma once

// Read or write the contents of a file in toto.

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace io {

// Note: catching std::ios_base::failure is broken for gcc versions before 7
// with C++11, owing to ABI issues.

struct bulkio_error: std::runtime_error {
    bulkio_error(std::string what): std::runtime_error(std::move(what)) {}
};

template <typename HasAssign>
void read_all(std::istream& in, HasAssign& A) {
    A.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

template <typename HasAssign>
void read_all(const std::string& filename, HasAssign& A) {
    try {
        std::ifstream fs;
        fs.exceptions(std::ios::failbit);
        fs.open(filename);
        read_all(fs, A);
    }
    catch (const std::exception&) {
        throw bulkio_error("failure reading "+filename);
    }
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
    try {
        std::ofstream fs;
        fs.exceptions(std::ios::failbit);
        fs.open(filename);
        write_all(data, fs);
    }
    catch (const std::exception&) {
        throw bulkio_error("failure writing "+filename);
    }
}

} // namespace io
