#include <cmath>

#include "lsystem.hpp"
#include "lsys_models.hpp"

static constexpr double inf = INFINITY;

// Predefined parameters for two classes of neurons. Numbers taken primarily
// from Ascoli et al. 2001, but some details (soma diameters for example)
// taken from earlier literature.
//
// Without mixture distribution support, these aren't entirely faithful to
// the original sources.
//
// Refer to `README.md` for references and more information.
// https://krasnow1.gmu.edu/cn3/L-Neuron/database/moto/bu/motoburk.prm
lsys_param make_alpha_motoneuron_lsys() {
    lsys_param L;

    // Soma diameter [µm].
    L.diam_soma = { 47.0, 65.5, 57.6, 5.0 }; // somadiam

    // Number of dendritic trees (rounded to nearest int).
    L.n_tree = { 8.0, 16.0 }; // numtree

    // Initial dendrite diameter [µm]. (Dstem)
    L.diam_initial = { 3.0, 12.0 }; // initdiam

    // Dendrite step length [µm]. (ΔL)
    L.length_step = { 25 };

    // Initial roll (intrinsic rotation about x-axis) [degrees].
    L.roll_initial = { -180.0, 180.0 };

    // Initial pitch (intrinsic rotation about y-axis) [degrees].
    L.pitch_initial = { 0.0, 180.0 }; // biforient

    // Tortuousness: roll within section over ΔL [degrees].
    L.roll_section = { -180.0, 180.0 };

    // Tortuousness: pitch within section over ΔL [degrees].
    L.pitch_section = { -15.0, 15.0, 0.0, 5.0 };

    // Taper rate: diameter decrease per unit length.
    L.taper = { -1.25e-3 };

    // Branching torsion: roll at branch point [degrees].
    L.roll_at_branch = { 0.0, 180.0 };

    // Branch angle between siblings [degrees].
    L.branch_angle = { 1.0, 172.0, 45.0, 20.0 }; // bifamplitude

    // Correlated child branch radius distribution parameters.
    L.diam_child_a = -0.2087;
    L.diam_child_r = { 0.2, 1.8, 0.8255, 0.2125 };

    // Termination probability parameters [1/µm].
    L.pterm_k1 = 2.62e-2;
    L.pterm_k2 = -2.955;

    // Bifurcation probability parameters [1/μm].
    L.pbranch_ov_k1 = 2.58e-5;
    L.pbranch_ov_k2 = 2.219;
    L.pbranch_nov_k1 = 2.34e-3;
    L.pbranch_nov_k2 = 0.194;

    // Absolute maximum dendritic extent [µm].
    L.max_extent = 2000;

    return L;
}

lsys_param alpha_motoneuron_lsys = make_alpha_motoneuron_lsys();

// Some parameters missing from the literature are taken from
// the `purkburk.prm` L-Neuron parameter file:
// http://krasnow1.gmu.edu/cn3/L-Neuron/database/purk/bu/purkburk.prm
// Note 'Pov' and 'Pterm' numbers are swapped in Ascoli fig 3B.

lsys_param make_purkinje_lsys() {
    lsys_param L;

    // Soma diameter [µm].
    // (Gaussian fit to 3 samples rom Rapp 1994); truncate to 2σ.
    L.diam_soma = { 22.12,  27.68, 24.9, 1.39 };

    // Number of dendritic trees (rounded to nearest int).
    L.n_tree = { 1 };

    // Initial dendrite diameter [µm]. (Dstem)
    L.diam_initial = { 4.8, 7.6, 6.167, 1.069 };

    // Dendrite step length [µm]. (ΔL)
    L.length_step = { 2.3 }; // from `purkburk.prm`

    // Tortuousness: roll within section over ΔL [degrees].
    L.roll_section = { 0 };

    // Tortuousness: pitch within section over ΔL [degrees].
    L.pitch_section = { -5, 5 }; // from `purkburk.prm`

    // Taper rate: diameter decrease per unit length.
    L.taper = { -inf, inf, -0.010, 0.022 };

    // Branching torsion: roll at branch point [degrees].
    L.roll_at_branch = { -inf, inf, 0.0, 6.5 }; // from `purkburj.prm`

    // Branch angle between siblings [degrees].
    L.branch_angle = { 1.0, 179.0, 71.0, 40.0 };

    // Correlated child branch radius distribution parameters.
    L.diam_child_a = 5.47078; // from `purkburk.prm`
    L.diam_child_r = { 0, inf, 0.112, 0.035 };

    // Termination probability parameters [1/µm].
    L.pterm_k1 = 0.1973; // from `purkburj.prm`; 0.4539 in Ascoli.
    L.pterm_k2 = -1.1533;

    // Bifurcation probability parameters [1/μm].
    L.pbranch_ov_k1 = 0.1021; // from `purkburk.prm`; 0.0355 in Ascoli.
    L.pbranch_ov_k2 = 0.7237;
    L.pbranch_nov_k1 = 0.1021; // from `purkburk.prm`; 0.2349 in Ascoli.
    L.pbranch_nov_k2 = -0.0116;

    // Absolute maximum dendritic extent [µm].
    L.max_extent = 20000;

    return L;
}

lsys_param purkinje_lsys = make_purkinje_lsys();


lsys_param make_apical_lsys() {
    lsys_param L;

    // Soma diameter [µm].
    L.diam_soma = { 47.0, 65.5, 57.6, 5.0 }; // somadiam

    // Number of dendritic trees (rounded to nearest int).
    L.n_tree = { 1 }; // numtree

    // Initial dendrite diameter [µm]. (Dstem)
    L.diam_initial = { 10, 20 }; // initdiam

    // Dendrite step length [µm]. (ΔL)
    L.length_step = { 25 };

    // Initial roll (intrinsic rotation about x-axis) [degrees].
    L.roll_initial = { 0 };

    // Initial pitch (intrinsic rotation about y-axis) [degrees].
    L.pitch_initial = { 0 }; // biforient

    // Tortuousness: roll within section over ΔL [degrees].
    L.roll_section = { -180.0, 180.0 };

    // Tortuousness: pitch within section over ΔL [degrees].
    L.pitch_section = { -15.0, 15.0, 0.0, 5.0 };

    // Taper rate: diameter decrease per unit length.
    L.taper = { -1.25e-3 };

    // Branching torsion: roll at branch point [degrees].
    L.roll_at_branch = { 0.0, 180.0 };

    // Branch angle between siblings [degrees].
    L.branch_angle = { 1.0, 172.0, 45.0, 20.0 }; // bifamplitude

    // Correlated child branch radius distribution parameters.
    L.diam_child_a = -0.2087;
    L.diam_child_r = { 0.2, 1.8, 0.8255, 0.2125 };

    // Termination probability parameters [1/µm].
    L.pterm_k1 = 2.62e-2;
    L.pterm_k2 = -2.955;

    // Bifurcation probability parameters [1/μm].
    L.pbranch_ov_k1 = 0.58e-5;
    L.pbranch_ov_k2 = 2.219;
    L.pbranch_nov_k1 = 2.34e-3;
    L.pbranch_nov_k2 = 0.194;

    // Absolute maximum dendritic extent [µm].
    L.max_extent = 2000;

    L.tag = 4;

    return L;
}

lsys_param apical_lsys = make_apical_lsys();

lsys_param make_basal_lsys() {
    lsys_param L;

    // Soma diameter [µm].
    L.diam_soma = { 47.0, 65.5, 57.6, 5.0 }; // somadiam

    // Number of dendritic trees (rounded to nearest int).
    L.n_tree = { 1, 20, 6, 2}; // numtree

    // Initial dendrite diameter [µm]. (Dstem)
    L.diam_initial = { 3.0, 12.0 }; // initdiam

    // Dendrite step length [µm]. (ΔL)
    L.length_step = { 25 };

    // Initial roll (intrinsic rotation about x-axis) [degrees].
    L.roll_initial = { -180, 180 };

    // Initial pitch (intrinsic rotation about y-axis) [degrees].
    L.pitch_initial = { 140, 220 }; // biforient

    // Tortuousness: roll within section over ΔL [degrees].
    L.roll_section = { -180.0, 180.0 };

    // Tortuousness: pitch within section over ΔL [degrees].
    L.pitch_section = { -15.0, 15.0, 0.0, 5.0 };

    // Taper rate: diameter decrease per unit length.
    L.taper = { -1.25e-3 };

    // Branching torsion: roll at branch point [degrees].
    L.roll_at_branch = { 0.0, 180.0 };

    // Branch angle between siblings [degrees].
    L.branch_angle = { 1.0, 172.0, 45.0, 20.0 }; // bifamplitude

    // Correlated child branch radius distribution parameters.
    L.diam_child_a = -0.2087;
    L.diam_child_r = { 0.2, 1.8, 0.8255, 0.2125 };

    // Termination probability parameters [1/µm].
    L.pterm_k1 = 2.62e-2;
    L.pterm_k2 = -2.955;

    // Bifurcation probability parameters [1/μm].
    L.pbranch_ov_k1 = 1.58e-5;
    L.pbranch_ov_k2 = 2.219;
    L.pbranch_nov_k1 = 2.34e-3;
    L.pbranch_nov_k2 = 0.194;

    // Absolute maximum dendritic extent [µm].
    L.max_extent = 1000;

    L.tag = 3;

    return L;
}

lsys_param basal_lsys = make_basal_lsys();
