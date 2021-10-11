## Directory Structure
```
|- test\
    |- unit\
        |- test_contexts.py
        |- ...
    |- unit-distributed\
        |- test_contexts_arbmpi.py
        |- test_contexts_mpi4py.py
        |- ...
```

In subfolders `unit`/`unit_distributed`:
- `test_xxxs.py`: define `TestMyTestCase(unittest.TestCase)` classes with
  test methods

## Usage

* to run all tests:

```             
[mpiexec -n X] python -m unittest discover [-v] -s python
```

* to run pattern matched test file(s):

```             
[mpiexec -n X] python -m unittest discover [-v] -s python -p test_some_file.py
[mpiexec -n X] python -m unittest discover [-v] -s python -p test_some_*.py
```


## Adding new tests

1. In suitable folder `test/unit` (no MPI) or `test/unit_distributed` (MPI),
  create `test_xxxs.py` file
1. Create tests suitable for local and distributed
  testing, or mark with the appropriate `cases.skipIf(Not)Distributed` decorator

## Naming convention

- modules: `test_xxxs` (ending with `s` since module can consist of multiple classes)
- class(es): `TestXxxs` (ending with `s` since class can consist of multiple test functions)
- functions: `test_yyy`
