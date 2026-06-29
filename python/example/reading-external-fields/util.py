import arbor as A
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from pathlib import Path
import numpy as np

ex = 1
ey = 0
ez = 0


def plot_morphology(mrf, *, fg=None):
    tree = mrf.segment_tree
    cs = cm.viridis
    fg = plt.figure(figsize=(10, 10))
    ax = fg.add_subplot(projection="3d")

    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    cmin = 1000
    cmax = -1000
    for seg in tree.segments:
        x0 = seg.prox.x
        x1 = seg.dist.x
        y0 = seg.prox.y
        y1 = seg.dist.y
        z0 = seg.prox.z
        z1 = seg.dist.z
        r0 = seg.prox.radius
        r1 = seg.dist.radius
        c = ((x1 - x0) * ex + (y1 - y0) * ey + (z1 - z0) * ez) / np.sqrt(
            (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) + (z1 - z0) * (z1 - z0)
        )
        cmin = min(cmin, c)
        cmax = max(cmax, c)
        c = cs(c)
        ax.plot(
            xs=[x0, x1], ys=[y0, y1], zs=[z0, z1], color=c, lw=(r0 + r1)
        )  # average diameter from radius
    ax.set_xlabel(r"x $(\mu m)$")
    ax.set_ylabel(r"y $(\mu m)$")
    ax.set_zlabel(r"z $(\mu m)$")
    print(cmin, cmax)
    fg.colorbar(cm.ScalarMappable(norm=norm, cmap=cs), ax=ax, shrink=0.5)
    fg.tight_layout()
    return fg, ax


here = Path(__file__).parent
mrf = A.load_swc_neuron(here / "Acker2008.swc")

fg, ax = plot_morphology(mrf)
fg.savefig("external-fields-morph.svg")
