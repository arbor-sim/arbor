#include <cmath>

#include <arbor/cable_cell.hpp>
#include <arbor/segment.hpp>
#include <arbor/mechinfo.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/recipe.hpp>

namespace arb {

/*
 * Create cell with just a soma:
 *
 * Soma:
 *    diameter: 18.8 µm
 *    mechanisms: HH (default params)
 *    bulk resistivitiy: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Stimuli:
 *    soma centre, t=[10 ms, 110 ms), 0.1 nA
 */

struct soma_cell_builder {
    double soma_rad;
    sample_tree tree;
    int ndend = 0;
    std::vector<int> branch_dist;
    std::vector<std::pair<mlocation, i_clamp>> stims;
    std::unordered_map<std::string, region> regions = {
        {"soma", reg::tagged(1)},
        {"axon", reg::tagged(2)},
        {"dend", reg::tagged(3)},
        {"apic", reg::tagged(4)},
    };

    std::unordered_multimap<std::string, mechanism_desc> mechanisms = {
        {"soma", "hh"},
        {"axon", "pas"},
        {"dend", "pas"},
        {"apic", "pas"},
    };

    std::unordered_multimap<std::string, cable_cell_local_parameter_set> properties;

    soma_cell_builder(double r): soma_rad(r) {
        tree.append({{0,0,0,r}, 1});
        branch_dist.push_back(0);
    }

    void add_branch(msize_t pb, double len, double r1, double r2, int ncomp, int tag) {
        auto p = branch_dist[pb];
        auto z = pb? tree.samples()[p].loc.z: soma_rad;

        p = tree.append(p, {{0,0,z,r1}, tag});
        if (ncomp>1) {
            auto dz = len/ncomp;
            auto dr = (r2-r1)/ncomp;
            for (auto i=1; i<ncomp; ++i) {
                p = tree.append(p, {{0,0,z+i*dz, r1+i*dr}, tag});
            }
        }
        p = tree.append(p, {{0,0,z+len,r2}, tag});
        branch_dist.push_back(p);
    }

    void add_dendrite(msize_t pb, double len, double r1, double r2, int ncomp) {
        add_branch(pb, len, r1, r2, ncomp, 3);
    }

    void add_stim(mlocation loc, i_clamp stim) {
        stims.push_back({loc, stim});
    }

    void set_regions(std::unordered_map<std::string, region> regs) {
        regions = std::move(regs);
    }

    void set_properties(std::unordered_multimap<std::string, cable_cell_local_parameter_set> props) {
        properties = std::move(props);
    }

    void set_mechanisms(std::unordered_multimap<std::string, mechanism_desc> mechs) {
        mechanisms = std::move(mechs);
    }

    cable_cell make_cell() const {
        // make dictionary
        label_dict dict;
        for (auto& reg: regions) {
            dict.set(reg.first, reg.second);
        }

        // make cable_cell from sample tree and dictionary
        cable_cell c(tree, dict, true);

        // add mechanisms
        for (auto& m: mechanisms) {
            c.paint(m.first, m.second);
        }

        // add properties
        for (auto& p: properties) {
            c.paint(p.first, p.second);
        }

        // add stimuli
        for (auto& s: stims) {
            c.place(s.first, s.second);
        }

        return c;
    }
};

inline cable_cell make_cell_soma_only(bool with_stim = true) {
    soma_cell_builder builder(18.8/2.0);
    if (with_stim) {
        builder.add_stim(mlocation{0,0.5}, i_clamp{10., 100., 0.1});
    }
    return builder.make_cell();
}

/*
 * Create cell with a soma and unbranched dendrite:
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: passive (default params)
 *    diameter: 1 µm
 *    length: 200 µm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

inline cable_cell make_cell_ball_and_stick(bool with_stim = true) {
    auto builder = soma_cell_builder(12.6157/2.0);
    builder.add_dendrite(0, 200, 1.0/2, 1.0/2, 4);
    if (with_stim) {
        builder.add_stim(mlocation{1,1}, i_clamp{5, 80, 0.3});
    }
    return builder.make_cell();
}

/*
 * Create cell with a soma and unbranched tapered dendrite:
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: passive (default params)
 *    diameter proximal: 1 µm
 *    diameter distal: 0.4 µm
 *    length: 200 µm
 *    bulk resistivity: 100 Ω·cm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

    /*
inline cable_cell make_cell_ball_and_taper(bool with_stim = true) {
  //cable_cell c;

  //auto soma = c.add_soma(12.6157/2.0);
  //soma->add_mechanism("hh");

  //c.add_cable(0, make_segment<cable_segment>(section_kind::dendrite, 1.0/2, 0.4/2, 200.0));

  //for (auto& seg: c.segments()) {
  //    if (seg->is_dendrite()) {
  //        seg->add_mechanism("pas");
  //        seg->set_compartments(4);
  //    }
  //}

  //if (with_stim) {
  //    c.place(mlocation{1,1}, i_clamp{5., 80., 0.3});
  //}
  //return c;

    auto builder = soma_cell_builder(12.6157/2.0);
    builder.add_dendrite(0, 200, 1.0/2, 0.4/2, 4);
    if (with_stim) {
        builder.add_stim(mlocation{1,1}, i_clamp{5, 80, 0.3});
    }
    return builder.make_cell({"soma", "hh"), {"dend", "pas"};
}
    */

/*
 * Create cell with a soma and unbranched dendrite with varying diameter:
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: passive (default params)
 *    length: 100 µm
 *    membrane resistance: 100 Ω·cm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

/*
inline cable_cell make_cell_ball_and_squiggle(bool with_stim = true) {
    cable_cell c;

    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism("hh");

    std::vector<cable_cell::value_type> radii;
    std::vector<cable_cell::point_type> points;

    double length = 100.0;
    int npoints = 200;

    for (int i=0; i<npoints; ++i) {
        double x = i*(1.0/(npoints-1));
        double r = std::exp(-x)*(std::sin(40*x)*0.05+0.1)+0.1;

        radii.push_back(r);
        points.push_back({x*length, 0., 0.});
    };

    auto dendrite =
        make_segment<cable_segment>(section_kind::dendrite, radii, points);
    c.add_cable(0, std::move(dendrite));

    for (auto& seg: c.segments()) {
        if (seg->is_dendrite()) {
            seg->add_mechanism("pas");
            seg->set_compartments(4);
        }
    }

    if (with_stim) {
        c.place(mlocation{1,1}, i_clamp{5., 80., 0.3});
    }
    return c;
}
*/

/*
 * Create cell with a soma and three-segment dendrite with single branch point:
 *
 * O----======
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrites:
 *    mechanisms: passive (default params)
 *    diameter: 1 µm
 *    length: 100 µm
 *    compartments: 4
 *
 * Stimulus:
 *    end of first terminal branch, t=[5 ms, 85 ms), 0.45 nA
 *    end of second terminal branch, t=[40 ms, 50 ms), -0.2 nA
 */

inline cable_cell make_cell_ball_and_3stick(bool with_stim = true) {
    auto builder = soma_cell_builder(12.6157/2.0);
    builder.add_dendrite(0, 100, 0.5, 0.5, 4);
    builder.add_dendrite(1, 100, 0.5, 0.5, 4);
    builder.add_dendrite(1, 100, 0.5, 0.5, 4);
    if (with_stim) {
        builder.add_stim(mlocation{2,1}, i_clamp{5.,  80., 0.45});
        builder.add_stim(mlocation{3,1}, i_clamp{40., 10.,-0.2});
    }
    return builder.make_cell();
}

/*
 * Create 'soma-less' cell with single cable, with physical
 * parameters from Rallpack 1 model.
 *
 * Common properties:
 *    mechanisms: passive
 *        membrane conductance: 0.000025 S/cm² ( =  1/(4Ω·m²) )
 *        membrane reversal potential: -65 mV (default)
 *    diameter: 1 µm
 *    length: 1000 µm
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[0 ms, inf), 0.1 nA
 *
 * Note: zero-volume soma added with same mechanisms, as
 * work-around for some existing fvm modelling issues.
 */

/*
inline cable_cell make_cell_simple_cable(bool with_stim = true) {
    cable_cell c;

    c.default_parameters.axial_resistivity = 100;
    c.default_parameters.membrane_capacitance = 0.01;

    c.add_soma(0);
    c.add_cable(0, make_segment<cable_segment>(section_kind::dendrite, 0.5, 0.5, 1000));

    double gbar = 0.000025;
    double I = 0.1;

    mechanism_desc pas("pas");
    pas["g"] = gbar;

    for (auto& seg: c.segments()) {
        seg->add_mechanism(pas);

        if (seg->is_dendrite()) {
            seg->set_compartments(4);
        }
    }

    if (with_stim) {
        // stimulus in the middle of our zero-volume 'soma'
        // corresponds to proximal end of cable.
        c.place(mlocation{0,0.5}, i_clamp{0., INFINITY, I});
    }
    return c;
}
*/
} // namespace arb
