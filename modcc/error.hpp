#pragma once

#include <deque>
#include <iterator>
#include <stdexcept>
#include <string>

#include "location.hpp"

struct error_entry {
    std::string message;
    Location location;

    error_entry(std::string m): message(std::move(m)) {}
    error_entry(std::string m, Location l): message(std::move(m)), location(l) {}
};

// Mixin class for managing a stack of error info.

class error_stack {
private:
    std::deque<error_entry> errors_;
    std::deque<error_entry> warnings_;

public:
    bool has_error() const { return !errors_.empty(); }
    void error(error_entry info) { errors_.push_back(std::move(info)); }
    void clear_errors() { errors_.clear(); }

    std::deque<error_entry>& errors() { return errors_; }
    const std::deque<error_entry>& errors() const { return errors_; }

    template <typename Seq>
    void append_errors(const Seq& seq) {
        errors_.insert(errors_.end(), std::begin(seq), std::end(seq));
    }

    bool has_warning() const { return !warnings_.empty(); }
    void warning(error_entry info) { warnings_.push_back(std::move(info)); }
    void clear_warnings() { warnings_.clear(); }

    std::deque<error_entry>& warnings() { return warnings_; }
    const std::deque<error_entry>& warnings() const { return warnings_; }

    template <typename Seq>
    void append_warnings(const Seq& seq) {
        warnings_.insert(warnings_.end(), std::begin(seq), std::end(seq));
    }
};

// Wrap error entry in exception.

class compiler_exception : public std::exception {
public:
    explicit compiler_exception(error_entry info)
    :   error_info_(std::move(info))
    {}

    compiler_exception(std::string m, Location location)
    :   error_info_({std::move(m), location})
    {}

    virtual const char* what() const throw() {
        return error_info_.message.c_str();
    }

    Location const& location() const {
        return error_info_.location;
    }

private:
    error_entry error_info_;
};
