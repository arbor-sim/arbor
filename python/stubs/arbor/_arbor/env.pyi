"""
Wrappers for arborenv.
"""
from __future__ import annotations
import arbor._arbor
import typing
__all__ = ['default_allocation', 'default_concurrency', 'default_gpu', 'find_private_gpu', 'get_env_num_threads', 'thread_concurrency']
def default_allocation() -> arbor._arbor.proc_allocation:
    """
    Attempts to detect the number of locally available CPU cores. Returns 1 if unable to detect the number of cores. Use with caution in combination with MPI.
    """
def default_concurrency() -> arbor._arbor.proc_allocation:
    """
    Returns number of threads to use from get_env_num_threads(), or else from thread_concurrency() if get_env_num_threads() returns zero.
    """
def default_gpu() -> int | None:
    """
    Determine GPU id to use from the ARBENV_GPU_ID environment variable, or from the first available GPU id of those detected.
    """
def find_private_gpu(arg0: typing.Any) -> None:
    """
    Identify a private GPU id per node, only available if built with GPU and MPI.
      mpi:     The MPI communicator.
    """
def get_env_num_threads() -> int:
    """
    Retrieve user-specified number of threads to use from the environment variable ARBENV_NUM_THREADS.
    """
def thread_concurrency() -> int:
    """
    Attempts to detect the number of locally available CPU cores. Returns 1 if unable to detect the number of cores. Use with caution in combination with MPI.
    """
