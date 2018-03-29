#pragma once

#include <string>

namespace arb {

/*
  Ion channels have the following fields, whose label corresponds to that
  in NEURON. We give them more easily understood accessors.

    ---------------------------------------------------
    label   Ca      Na      K   name
    ---------------------------------------------------
    iX      ica     ina     ik  current_density
    eX      eca     ena     ek  reversal_potential
    Xi      cai     nai     ki  internal_concentration
    Xo      cao     nao     ko  external_concentration
    gX      gca     gna     gk  conductance
    ---------------------------------------------------
*/

// Fixed set of ion species (to be generalized in the future):

enum class ionKind {ca, na, k};
inline std::string to_string(ionKind k) {
    switch (k) {
    case ionKind::ca: return "ca";
    case ionKind::na: return "na";
    case ionKind::k:  return "k";
    default: throw std::out_of_range("unknown ionKind");
    }
}

// Ion (species) description

struct ion_info {
    ionKind kind;
    int charge; // charge of ionic species
    double default_int_concentration; // (mM) default internal concentration
    double default_ext_concentration; // (mM) default external concentration
};

} // namespace arb

