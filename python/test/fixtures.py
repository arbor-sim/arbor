import arbor
import functools
from functools import lru_cache as cache
import unittest
from pathlib import Path
import subprocess
import warnings

_mpi_enabled = arbor.__config__["mpi"]
_mpi4py_enabled = arbor.__config__["mpi4py"]

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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs[param_name] = fixture()
        return func(*args, **kwargs)

    return wrapper

def _fixture(decorator):
    @functools.wraps(decorator)
    def fixture_decorator(func):
        return _fix(decorator.__name__, decorator, func)

    return fixture_decorator

def _singleton_fixture(f):
    return _fixture(cache(f))


@_fixture
def repo_path():
    """
    Fixture that returns the repo root path.
    """
    return Path(__file__).parent.parent.parent


@_fixture
def context():
    """
    Fixture that produces an MPI sensitive `arbor.context`
    """
    args = [arbor.proc_allocation()]
    if _mpi_enabled:
        if not arbor.mpi_is_initialized():
            print("Context fixture initializing mpi", flush=True)
            arbor.mpi_init()
        if _mpi4py_enabled:
            from mpi4py.MPI import COMM_WORLD as comm
        else:
            comm = arbor.mpi_comm()
        args.append(comm)
    return arbor.context(*args)


class _BuildCatError(Exception): pass


def _build_cat_local(name, path):
    try:
        subprocess.run(["build-catalogue", name, str(path)], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise _BuildCatError("Tests can't build catalogues:\n" + e.stderr.decode()) from None


def _build_cat_distributed(comm, name, path):
    # Control flow explanation:
    # * `build_err` starts out as `None`
    # * Rank 1 to N wait for a broadcast from rank 0 to receive the new value
    #   for `build_err`
    # * Rank 0 splits off from the others and executes the build.
    #   * If it builds correctly it finishes the collective `build_err`
    #     broadcast with the initial value `None`: all nodes continue.
    #   * If it errors, it finishes the collective broadcast with the caught err
    # * All MPI ranks either continue or raise the same err. (prevents stalling)
    build_err = None
    try:
        if not comm.Get_rank():
            _build_cat_local(name, path)
        build_err = comm.bcast(build_err, root=0)
    except Exception as e:
        build_err = comm.bcast(e, root=0)
    if build_err:
        raise build_err

@context
def _build_cat(name, path, context):
    if context.has_mpi:
        try:
            from mpi4py.MPI import COMM_WORLD as comm
            
            serial = False
        except ImportError:
            warnings.warn(
                "Building catalogue in an MPI context, but `mpi4py` not found."
                + " Concurrent identical catalogue builds might occur."
            )
            serial = True
    else:
        serial = True
    if serial:
        _build_cat_local(name, path)
    else:
        _build_cat_distributed(comm, name, path)
    return Path.cwd() / (name + "-catalogue.so")


@_singleton_fixture
@repo_path
def dummy_catalogue(repo_path):
    """
    Fixture that returns a dummy `arbor.catalogue`
    which contains the `dummy` mech.
    """
    path = repo_path / "test" / "unit" / "dummy"
    cat_path = _build_cat("dummy", path)
    return arbor.load_catalogue(str(cat_path))

@_fixture
class empty_recipe(arbor.recipe):
    """
    Fixture that returns a blank `arbor.recipe` instance.
    """
    pass
