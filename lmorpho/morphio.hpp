#pragma once

#include <fstream>
#include <iostream>
#include <vector>

#include <arbor/morphology.hpp>

// Manage access to a single file, std::cout, or an indexed
// sequence of files.

class multi_file {
private:
    std::ofstream file_;
    bool concat_ = false;
    bool use_stdout_ = false;
    // use if not concat_:
    std::string fmt_prefix_;
    std::string fmt_suffix_;
    int fmt_digits_ = 0;
    // use if concat_:
    std::string filename_;   // use if concat_
    unsigned current_n_ = 0;

public:
    // If pattern is '-' or empty, represent std::cout.
    // If pattern contains '%', use a different file for each index.
    // Otherwise, use a single file given by pattern.
    explicit multi_file(const std::string& pattern, int digits=0);

    multi_file(multi_file&&) = default;

    // Open file for index n, closing previously open file if different.
    void open(unsigned n);

    // Close currently open file.
    void close() { if (!use_stdout_ && file_.is_open()) file_.close(); }

    // True if writing to a single file or std::cout.
    bool single_file() const { return concat_; }

    // Return std::cout or file stream as appropriate.
    std::ostream& stream() { return use_stdout_? std::cout: file_; }
};

static constexpr unsigned digits(unsigned n) {
    return n? 1+digits(n/10): 0;
}

// Write a sequence of morphologies to one or more files
// as given my `pattern`.

class swc_emitter {
    multi_file file_;

public:
    // `pattern` is as for `multi_file`; number `n` optionally
    // specifies the total number of morphologies, for better
    // numeric formatting.
    explicit swc_emitter(std::string pattern, unsigned n=0):
        file_(pattern, digits(n)) {}

    swc_emitter(swc_emitter&&) = default;

    // write `index`th morphology as SWC. 
    void operator()(unsigned index, const arb::morphology& m);

    void close() { file_.close(); }
    ~swc_emitter() { close(); }
};

// Write pvectors for a sequence of morphologies to one or more files
// as given my `pattern`. Coalesce the vectors if writing to a single
// file/stream.

class pvector_emitter {
    multi_file file_;
    bool coalesce_ = false;
    unsigned offset_ = 0;

public:
    // `pattern` is as for `multi_file`; number `n` optionally
    // specifies the total number of morphologies, for better
    // numeric formatting.
    explicit pvector_emitter(std::string pattern, unsigned n=0):
        file_(pattern, digits(n)),
        coalesce_(file_.single_file()) {}

    pvector_emitter(pvector_emitter&&) = default;

    // write pvector for `index`th morphology. 
    void operator()(unsigned index, const arb::morphology& m);

    void close() { file_.close(); }
    ~pvector_emitter() { close(); }
};

