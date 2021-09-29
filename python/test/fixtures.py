import arbor
import functools
from functools import lru_cache as cache
import unittest
from pathlib import Path
import subprocess

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


def _build_cat(name, path):
    from mpi4py.MPI import COMM_WORLD as comm
    build_err = None
    try:
        if not comm.Get_rank():
            subprocess.run(["build-catalogue", name, str(path / "uhsahh")], check=True, capture_output=True)
        build_err = comm.bcast(build_err, root=0)
    except Exception as e:
        build_err = comm.bcast(e, root=0)
    if build_err:
        raise RuntimeError("Tests can't build catalogues:\n" + build_err.stderr.decode())
    return Path.cwd() / (name + "-catalogue.so")


@_singleton_fixture
@repo_path
def dummy_catalogue(repo_path):
    path = repo_path / "test" / "unit" / "dummy"
    cat_path = _build_cat("dummy", path)
    return arbor.load_catalogue(str(cat_path))

@_fixture
class empty_recipe(arbor.recipe):
    pass
