#pragma once

#include "location.hpp"

class compiler_exception : public std::exception {
public:
    compiler_exception(std::string m, Location location)
    :   location_(location),
        message_(std::move(m))
    {}

    virtual const char* what() const throw() {
        return message_.c_str();
    }

    Location const& location() const {
        return location_;
    }

private:

    Location location_;
    std::string message_;
};

