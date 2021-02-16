#include <iostream>
#include <fstream>

#include <arborio/neurolucida.hpp>

#include <optional>

namespace arborio {

asc_no_document::asc_no_document():
    asc_exception("no neurolucida asc file to parse")
{}

asc_morphology load_asc(std::string filename) {
    std::ifstream fid(filename);

    std::cout << "loading " << filename << "\n";
    if (!fid.good()) {
        throw asc_no_document();
    }

    std::string fstr; // will hold contents of input file
    fid.seekg(0, std::ios::end);
    fstr.reserve(fid.tellg());
    fid.seekg(0, std::ios::beg);

    fstr.assign((std::istreambuf_iterator<char>(fid)),
                 std::istreambuf_iterator<char>());



    return {};
}

} // namespace arborio
