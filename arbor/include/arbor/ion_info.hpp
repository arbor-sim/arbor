#pragma once

namespace arb {

struct ion_info {
    int charge;                       // charge of ionic species
    double default_int_concentration; // (mM) default internal concentration
    double default_ext_concentration; // (mM) default external concentration
};

} // namespace arb
