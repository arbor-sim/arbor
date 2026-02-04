import arbor as A
from arbor import units as U
import functools
from functools import lru_cache as cache
from pathlib import Path
import subprocess
import inspect

# The API of `functools`'s caches went through a bunch of breaking changes from
# 3.6 to 3.9. Patch them up in a local `cache` function.
try:
    cache(lambda: None)
except TypeError:
    # If `lru_cache` does not accept user functions as first arg, it expects
    # the max cache size as first arg, we pass None to produce a cache decorator
    # without max size.
    cache = cache(None)


def _fix(param_name, fixture, func):
    """
    Decorates `func` to inject the `fixture` callable result as `param_name`.
    """
    sig = inspect.signature(func)
    if param_name not in sig.parameters:
        raise TypeError(
            f"{param_name} fixture can't be applied to a function without {param_name}"
            " parameter"
        )

    @functools.wraps(func)
    def inject_fixture(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        if param_name not in bound.arguments:
            bound.arguments[param_name] = fixture
        return func(*bound.args, **bound.kwargs)

    return inject_fixture


def _fixture(fixture_factory):
    """
    Takes a fixture factory and returns a decorator factory, so that when the fixture
    factory is called, a decorator is produced that injects the fixture values when the
    decorated function is called.
    """

    @functools.wraps(fixture_factory)
    def decorator_fatory(*args, **kwargs):
        def decorator(func):
            fixture = fixture_factory(*args, **kwargs)
            return _fix(fixture_factory.__name__, fixture, func)

        return decorator

    return decorator_fatory


def _singleton_fixture(f):
    return _fixture(cache(f))


@_fixture
def repo_path():
    """
    Fixture that returns the repo root path.
    """
    return Path(__file__).parent.parent.parent


def get_mpi_comm_world():
    """
    Obtain MPI_COMM_WORLD as --- in order ---
    1. MPI4PY.MPI.COMM_WORLD
    2. Arbor MPI
    3. None
    """
    if A.config()["mpi"]:
        if A.config()["mpi4py"]:
            from mpi4py import MPI

            return MPI.COMM_WORLD
        else:
            if not A.mpi_is_initialized():
                A.mpi_init()
            return A.mpi_comm()
    return None


@_fixture
def context():
    """
    Fixture that produces an MPI sensitive `A.context`
    """
    return A.context(mpi=get_mpi_comm_world())


@_fixture
def single_context():
    """
    Fixture that produces an non-MPI `A.context`, which includes
    a GPU, if enabled.
    """
    mpi = None
    gpu = None
    if A.config()["gpu"]:
        gpu = 0
    return A.context(mpi=mpi, gpu_id=gpu)


class _BuildCatError(Exception):
    pass


def _build_cat_local(name, path):
    try:
        subprocess.run(
            ["arbor-build-catalogue", name, str(path)],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise _BuildCatError(
            f"Tests can't build catalogue '{name}' from '{path}':\n"
            f"{e.stderr.decode()}\n\n{e.stdout.decode()}"
        ) from None


def _build_cat_distributed(comm, name, path):
    # Control flow explanation:
    # * `build_err` starts out as `None`
    # * Rank 1 to N wait for a broadcast from rank 0 to receive the new value
    #   for `build_err`
    # * Rank 0 splits off from the others and executes the build.
    #   * If it builds correctly it finishes the collective `build_err`
    #     broadcast with the initial value `None`: all nodes continue.
    #   * If it errors, it finishes the collective broadcast with the caught err
    #
    # All MPI ranks either continue or raise the same err. (prevents stalling)
    build_err = None
    if not comm.Get_rank():
        try:
            _build_cat_local(name, path)
        except Exception as e:
            build_err = e
    build_err = comm.bcast(build_err, root=0)
    if build_err:
        raise build_err


@context()
def _build_cat(name, path, context):
    if context.has_mpi:
        try:
            from mpi4py.MPI import COMM_WORLD as comm
        except ImportError:
            raise _BuildCatError(
                "Building catalogue in an MPI context, but `mpi4py` not found."
                " Concurrent identical catalogue builds might occur."
            ) from None

        _build_cat_distributed(comm, name, path)
    else:
        _build_cat_local(name, path)
    return Path.cwd() / (name + "-catalogue.so")


@_singleton_fixture
@repo_path()
def dummy_catalogue(repo_path):
    """
    Fixture that returns a dummy `A.catalogue`
    which contains the `dummy` mech.
    """
    path = repo_path / "test" / "unit" / "dummy"
    cat_path = _build_cat("dummy", path)
    return A.load_catalogue(str(cat_path))


@_singleton_fixture
@repo_path()
def diffusion_catalogue(repo_path):
    """
    Fixture that returns an `arbor.catalogue`
    which contains mechanisms `neuron_with_diffusion`
    and `synapse_with_diffusion`.
    """
    path = repo_path / "test" / "unit" / "diffusion"
    cat_path = _build_cat("diffusion", path)
    return A.load_catalogue(str(cat_path))


@_fixture
class empty_recipe(A.recipe):
    """
    Blank recipe fixture.
    """

    pass


class art_spiker_recipe(A.recipe):
    """
    Recipe fixture with 3 artificial spiking cells and one cable cell.
    """

    def __init__(self):
        super().__init__()
        self.the_props = A.neuron_cable_properties()
        self.trains = [[0.8, 2, 2.1, 3], [0.4, 2, 2.2, 3.1, 4.5], [0.2, 2, 2.8, 3]]

        tree, labels, decor = self._cable_cell_elements()
        self.the_cell = A.cable_cell(tree, decor, labels)

    def _cable_cell_elements(self):
        # (1) Create a morphology with a single (cylindrical) segment of length=diameter
        #  = # 6 μm
        tree = A.segment_tree()
        tree.append(
            A.mnpos,
            (-3, 0, 0, 3),
            (3, 0, 0, 3),
            tag=1,
        )

        # (2) Define the soma and its midpoint
        labels = A.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

        # (3) Create cell and set properties
        decor = (
            A.decor()
            .set_property(Vm=-40 * U.mV)
            .paint('"soma"', A.density("hh"))
            .place('"midpoint"', A.iclamp(10 * U.ms, 2 * U.ms, 0.8 * U.nA), "iclamp")
            .place('"midpoint"', A.threshold_detector(-10 * U.mV), "detector")
        )

        # return tuple of tree, labels, and decor for creating a cable cell (can still
        # be modified before calling A.cable_cell())
        return tree, labels, decor

    def num_cells(self):
        return 4

    def cell_kind(self, gid):
        if gid >= 3:
            return A.cell_kind.cable
        return A.cell_kind.spike_source

    def global_properties(self, kind):
        if kind == A.cell_kind.cable:
            return self.the_props
        return None

    def probes(self, gid):
        if gid >= 3:
            return [A.cable_probe_membrane_voltage('"midpoint"', "Um")]
        return []

    def cell_description(self, gid):
        if gid >= 3:
            return self.the_cell
        return A.spike_source_cell("src", self.schedule(gid))

    def schedule(self, gid):
        return A.explicit_schedule([t * U.ms for t in self.trains[gid]])


@_fixture
def sum_weight_hh_spike():
    """
    Fixture returning connection weight for 'expsyn_stdp' mechanism which is just enough
    to evoke an immediate spike at t=1ms in the 'hh' neuron in 'art_spiker_recipe'
    """
    return 0.4


@_fixture
def sum_weight_hh_spike_2():
    """
    Fixture returning connection weight for 'expsyn_stdp' mechanism which is just enough
    to evoke an immediate spike at t=1.8ms in the 'hh' neuron in 'art_spiker_recipe'
    """
    return 0.36


@_fixture
@context()
def art_spiking_sim(context):
    rec = art_spiker_recipe()
    dd = A.partition_load_balance(rec, context)
    return A.simulation(rec, context, dd)
