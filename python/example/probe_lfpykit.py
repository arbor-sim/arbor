#!/usr/bin/env python

# Example utilizing the **`LFPykit`** module (https://lfpykit.readthedocs.io,
# https://github.com/LFPy/LFPykit) for predictions of extracellular
# potentials using the line source approximation implementation
# `LineSourcePotential` with a passive neuron model set up in Arbor
# (https://arbor.readthedocs.io, https://github.com/arbor-sim/arbor).
#
# The neuron receives sinusoid synaptic current input in one arbitrary
# chosen control volume (CV).
# Its morphology is defined in the file `single_cell_detailed.swc`

# import modules
import sys
import numpy as np
import arbor as A
from arbor import units as U
import lfpykit
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path


class Recipe(A.recipe):
    def __init__(self, cell):
        super().__init__()

        self.the_cell = cell

        self.vprobeset_id = (0, 0)
        self.iprobeset_id = (0, 1)
        self.cprobeset_id = (0, 2)

    def num_cells(self):
        return 1

    def cell_kind(self, _):
        return A.cell_kind.cable

    def cell_description(self, _):
        return self.the_cell

    def global_properties(self, _):
        return A.neuron_cable_properties()

    def probes(self, _):
        return [
            A.cable_probe_membrane_voltage_cell("Um-all"),
            A.cable_probe_total_current_cell("Itotal-all"),
            A.cable_probe_stimulus_current_cell("Istim-all"),
        ]


# Read the SWC filename from input
if len(sys.argv) == 1:
    print("No SWC file passed to the program, using default.")
    filename = Path(__file__).parent / "single_cell_detailed.swc"
elif len(sys.argv) == 2:
    filename = Path(sys.argv[1])
else:
    print("Usage: single_cell_detailed.py [SWC file name]")
    sys.exit(1)

morphology = A.load_swc_arbor(filename).morphology

# define a location on morphology for current clamp
clamp_location = A.location(4, 1 / 6)

# define a sinusoid input current
iclamp = A.iclamp(
    5 * U.ms,  # stimulation onset
    1e8 * U.ms,  # stimulation duration
    -0.001 * U.nA,  # stimulation amplitude
    frequency=100 * U.Hz,  # stimulation frequency
    phase=0 * U.rad,  # stimulation phase
)

decor = (
    A.decor()
    # set initial voltage, temperature, axial resistivity, membrane capacitance
    .set_property(
        Vm=-65 * U.mV,  # Initial membrane voltage (mV)
        tempK=300 * U.Kelvin,  # Temperature (Kelvin)
        rL=10 * U.kOhm * U.cm,  # Axial resistivity (Ω cm)
        cm=0.01 * U.F / U.m2,  # Membrane capacitance (F/m**2)
    )
    # set passive mech w. leak reversal potential (mV)
    .paint("(all)", A.density("pas/e=-65", g=0.0001))
    # attach the stimulus
    .place(str(clamp_location), iclamp, "iclamp")
    # use a fixed 3 CVs per branch
    .discretization(A.cv_policy_fixed_per_branch(3))
)

# place_pwlin can be queried with region/locset expressions to obtain
# geometrical objects, like points and segments, essentially recovering
# geometry from morphology.
ppwl = A.place_pwlin(morphology)
cell = A.cable_cell(morphology, decor)

# instantiate recipe with cell
rec = Recipe(cell)

# instantiate simulation
sim = A.simulation(rec)

# set up sampling on probes with sampling every 1 ms
schedule = A.regular_schedule(1.0 * U.ms)
v_handle = sim.sample(0, "Um-all", schedule)
i_handle = sim.sample(0, "Itotal-all", schedule)
c_handle = sim.sample(0, "Istim-all", schedule)

# run simulation for 500 ms of simulated activity and collect results.
sim.run(tfinal=500 * U.ms)

# extract time, V_m, I_m and I_c for each CV
V_m_samples, V_m_meta = sim.samples(v_handle)[0]
I_m_samples, I_m_meta = sim.samples(i_handle)[0]
I_c_samples, I_c_meta = sim.samples(c_handle)[0]

# drop recorded V_m values and corresponding metadata of
# zero-sized CVs (branch-point potentials).
# Here this is done as the V_m values are only used for visualization purposes
inds = np.array([m.dist != m.prox for m in V_m_meta])
V_m_samples = V_m_samples[:, np.r_[True, inds]]
V_m_meta = np.array(V_m_meta)[inds].tolist()

# assert that the remaining cables comprising the metadata for each probe are
# identical, as well as the reported sample times.
assert V_m_meta == I_m_meta
assert (V_m_samples[:, 0] == I_m_samples[:, 0]).all()

# prep recorded data for plotting and computation of extracellular potentials
time = V_m_samples[:, 0]
V_m = V_m_samples[:, 1:]

# Add stimulation current to transmembrane current to mimic sinusoid synapse
# current embedded in the membrane.
I_m = I_c_samples[:, 1:] + I_m_samples[:, 1:]  # (nA)


###############################################################################
# Compute extracellular potentials
###############################################################################


# ## Compute extracellular potentials
# First we define a couple of classes to interface the LFPykit
# library (https://LFPykit.readthedocs.io, https://github.com/LFPy/LFPykit):
class ArborCellGeometry(lfpykit.CellGeometry):
    """
    Class inherited from  ``lfpykit.CellGeometry`` for easier forward-model
    predictions in Arbor that keeps track of A.segment information
    for each CV.

    Parameters
    ----------
    p: ``A.place_pwlin`` object
        3-d locations and cables in a morphology (cf. ``A.place_pwlin``)
    cables: ``list``
        ``list`` of corresponding ``A.cable`` objects where transmembrane
        currents are recorded (cf. ``A.cable_probe_total_current_cell``)

    See also
    --------
    lfpykit.CellGeometry
    """

    def __init__(self, p, cables):
        x, y, z, d = [np.array([], dtype=float).reshape((0, 2))] * 4
        CV_ind = np.array([], dtype=int)  # tracks which CV owns segment
        for i, m in enumerate(cables):
            segs = p.segments([m])
            for seg in segs:
                x = np.row_stack([x, [seg.prox.x, seg.dist.x]])
                y = np.row_stack([y, [seg.prox.y, seg.dist.y]])
                z = np.row_stack([z, [seg.prox.z, seg.dist.z]])
                d = np.row_stack([d, [seg.prox.radius * 2, seg.dist.radius * 2]])
                CV_ind = np.r_[CV_ind, i]

        super().__init__(x=x, y=y, z=z, d=d)
        self._CV_ind = CV_ind


class ArborLineSourcePotential(lfpykit.LineSourcePotential):
    """subclass of ``lfpykit.LineSourcePotential`` modified for
    instances of ``ArborCellGeometry``.
    Each CV may consist of several segments , and this implementation
    accounts for their contributions normalized by surface area, that is,
    we assume constant transmembrane current density per area across each CV
    and constant current source density per unit length per segment
    (inherent in the line-source approximation).

    Parameters
    ----------
    cell: object
        ``ArborCellGeometry`` instance or similar.
    x: ndarray of floats
        x-position of measurement sites (µm)
    y: ndarray of floats
        y-position of measurement sites (µm)
    z: ndarray of floats
        z-position of measurement sites (µm)
    sigma: float > 0
        scalar extracellular conductivity (S/m)

    See also
    --------
    lfpykit.LineSourcePotential
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._get_transformation_matrix = super().get_transformation_matrix

    def get_transformation_matrix(self):
        """Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (n_coords, n_CVs) ndarray
        """
        M_tmp = self._get_transformation_matrix()
        n_CVs = np.unique(self.cell._CV_ind).size
        M = np.zeros((self.x.size, n_CVs))
        for i in range(n_CVs):
            inds = self.cell._CV_ind == i
            M[:, i] = M_tmp[:, inds] @ (
                self.cell.area[inds] / self.cell.area[inds].sum()
            )

        return M


# create ``ArborCellGeometry`` instance
cell_geometry = ArborCellGeometry(ppwl, I_m_meta)

# define locations where extracellular potential is predicted in vicinity
# of cell.
# Axis limits [x-min, x-max, y-min, y-max] (µm)
axis = np.array([-110, 370, -80, 70])
dx = 2  # spatial resolution along x-axis (µm)
dz = 2  # spatial resolution along y-axis (µm)
X, Y = np.meshgrid(
    np.linspace(axis[0], axis[1], int(np.diff(axis[:2]) // dx) + 1),
    np.linspace(axis[2], axis[3], int(np.diff(axis[2:]) // dz) + 1),
)
Z = np.zeros_like(X)

# ``ArborLineSourcePotential`` instance, get mapping for all segments per CV
lsp = ArborLineSourcePotential(
    cell=cell_geometry, x=X.flatten(), y=Y.flatten(), z=Z.flatten()
)
M = lsp.get_transformation_matrix()

# Extracellular potential in x,y-plane (mV)
V_e = M @ I_m.T


###############################################################################
# Plotting
###############################################################################
# Plot the morphology and extracellular potential prediction.
# First we define a couple helper functions:
def create_polygon(x, y, d):
    """create an outline for each segment defined by 1D arrays `x`, `y`, `d`
    in x,y-plane which can be drawn using `plt.Polygon`

    Parameters
    ----------
    x: ndarray
    y: ndarray
    d: ndarray

    Returns
    -------
    x, y: nested list
    """
    x_grad = np.gradient(x)
    y_grad = np.gradient(y)
    theta = np.arctan2(y_grad, x_grad)

    xp = np.r_[
        (x + 0.5 * d * np.sin(theta)).ravel(),
        (x - 0.5 * d * np.sin(theta)).ravel()[::-1],
    ]
    yp = np.r_[
        (y - 0.5 * d * np.cos(theta)).ravel(),
        (y + 0.5 * d * np.cos(theta)).ravel()[::-1],
    ]

    return list(zip(xp, yp))


def get_cv_polycollection(cell_geometry, V_m, vlims=(-66, -64), cmap="viridis"):
    """
    Parameters
    ----------
    cell_geometry: ``ArborCellGeometry`` object
    V_m: ndarray
        membrane voltages at some time point
    vlims: list
        color limits
    cmap: str
        matplotlib colormap name

    Returns
    -------
    PolyCollection
    """
    norm = plt.Normalize(vmin=vlims[0], vmax=vlims[1], clip=True)
    colors = [plt.get_cmap(cmap)(norm(v)) for v in V_m]
    zips = []
    for i in range(V_m.size):
        inds = cell_geometry._CV_ind == i
        zips.append(
            create_polygon(
                cell_geometry.x[inds,].flatten(),
                cell_geometry.y[inds,].flatten(),
                cell_geometry.d[inds,].flatten(),
            )
        )
    polycol = PolyCollection(zips, edgecolors=colors, facecolors=colors, linewidths=0.0)
    return polycol


def get_segment_outlines(cell_geometry):
    """
    Parameters
    ----------
    cell_geometry: ``ArborCellGeometry`` object
    cmap: str
        matplotlib colormap name

    Returns
    -------
    PolyCollection
    """
    zips = []
    for x_, y_, d_ in zip(cell_geometry.x, cell_geometry.y, cell_geometry.d):
        zips.append(create_polygon(x_, y_, d_))
    polycol = PolyCollection(zips, edgecolors="k", facecolors="none", linewidths=0.5)
    return polycol


def colorbar(
    fig,
    ax,
    im,
    width=0.01,
    height=1.0,
    hoffset=0.01,
    voffset=0.0,
    orientation="vertical",
):
    """
    draw matplotlib colorbar without resizing the parent axes object
    """
    rect = np.array(ax.get_position().bounds)
    rect = np.array(ax.get_position().bounds)
    caxrect = [0] * 4
    caxrect[0] = rect[0] + rect[2] + hoffset * rect[2]
    caxrect[1] = rect[1] + voffset * rect[3]
    caxrect[2] = rect[2] * width
    caxrect[3] = rect[3] * height
    cax = fig.add_axes(caxrect)
    cax.grid(False)
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    return cb


# show predictions at the last time point of simulation
time_index = -1

# create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 6), dpi=100)

# plot contours of V_e
lim = float(f"{abs(V_e).max() / 3:.1e}")
levels = np.linspace(-lim, lim, 25)
im_V_e = ax.contourf(
    X, Y, V_e[:, time_index].reshape(X.shape), cmap="RdBu", levels=levels, extend="both"
)

# V_e colorbar:
cb = colorbar(fig, ax, im_V_e, height=0.45, voffset=0.55)
cb.set_label("$V_e$ (mV)")

# add outline of each CV with color coding according to membrane voltage
vlims = [-66.0, -64.0]
polycol = get_cv_polycollection(cell_geometry, V_m[time_index, :], vlims=vlims)
im_V_m = ax.add_collection(polycol)

# V_m colorbar
cb2 = colorbar(fig, ax, im_V_m, height=0.45)
cb2.set_ticks([0, 0.5, 1])
cb2.set_ticklabels([vlims[0], np.mean(vlims), vlims[1]])
cb2.set_label(r"$V_m$ (mV)")

# draw segment outlines
ax.add_collection(get_segment_outlines(cell_geometry))

# add marker denoting clamp location
point = ppwl.at(clamp_location)
ax.plot(point.x, point.y, "ko", ms=10, label="stimulus")

ax.legend()

# axis annotations
ax.axis(axis)
ax.set_xlabel(r"$x$ ($\mu$m)", labelpad=0)
ax.set_ylabel(r"$y$ ($\mu$m)", labelpad=0)
ax.set_title(f"$V_e$ and $V_m$ at $t$={time[time_index]} ms")

# save file
fig.savefig("single_cell_extracellular_potentials.svg", bbox_inches="tight")

# ## Notes on output:
# The spatial discretization is here deliberately coarse with only 3 CVs
# per branch.
# Hence the branch receiving input about 1/6 of the way from its root
# (from `decor.place('(location 4 0.16667)', iclamp, '"iclamp"')`) is treated
# as 3 separate line sources with inhomogeneous current density per length
# unit. This inhomogeneity is due to the fact that the total transmembrane
# current per CV may distributed across multiple segments with varying surface
# area. The transmembrane current is assumed to be constant per length unit
# per segment.
