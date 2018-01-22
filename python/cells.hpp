#include <cell.hpp>

/*
 * Create cell with just a soma:
 *
 * Soma:
 *    diameter: 18.8 µm
 *    mechanisms: HH (default params)
 *    bulk resistivitiy: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 */

inline arb::cell make_cell_soma_only() {
    arb::cell c;

    auto soma = c.add_soma(18.8/2.0);
    soma->add_mechanism("hh");

    return c;
}

