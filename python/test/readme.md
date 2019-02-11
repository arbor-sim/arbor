## Directory Structure
```
|- test\
    |- options.py
    |- unit\
        |- runner.py
        |- test_contexts.py
        |- ...
    |- unit-distributed\
        |- runner.py
        |- test_contexts_arbmpi.py
        |- test_contexts_mpi4py.py
        |- ...
```

In parent folder `test`: 
- `options.py`: set global options (define arg parser)

In subfolders `unit`/`unit_distributed`: 
- `test_xxxs.py`: define unittest class with test methods and own test suite (named: test module)
- `runner.py`: run all tests in this subfolder (which are defined as suite in test modules) 

## Usage
### with `unittest` from SUBFOLDER: 

* to run all tests in subfolder:  
```             
python -m unittest [-v]
```
* to run module: 
```  
python -m unittest module [-v]
```  
, e.g. in `test/unit` use `python -m unittest test_contexts -v`
* to run class in module: 
```
python -m unittest module.class [-v]
```  
, eg. in `test/unit` use `python -m unittest test_contexts.Contexts -v`
* to run method in class in module: 
```  
python -m unittest module.class.method [-v]
```  
, eg. in `test/unit` use `python -m unittest test_contexts.Contexts.test_context -v`

### with `runner.py` and argument(s) `-v {0,1,2}` from SUBFOLDER: 

* to run all tests in subfolder:   
```  
python -m runner[-v2]
```   
or `python runner.py [-v2]`
* to run module: 
```  
python -m test_xxxs [-v2]
```   
or `python test_xxxs.py [-v2]`
* running classes or methods not possible this way

### from any other folder: 

* to run all tests:   
```
python path/to/runner.py [-v2]
```
* to run module: 
```  
python path/to/test_xxxs.py [-v2]
```   

## Adding new tests

1. In suitable folder `test/unit` (no MPI) or `test/unit_distributed` (MPI), create `test_xxxs.py` file
2. In  `test_xxxs.py` file, define 
  a) a unittest `class Xxxs(unittest.TestCase)` with test methods `test_yyy` 
  b) a suite function `suite()` consisting of all desired tests returning a unittest suite `unittest.makeSuite(Xxxs, ('test'))` (for all defined tests, tuple of selected tests possible); steering of which tests to include happens here!
  c) a run function `run()` with a unittest runner `unittest.TextTestRunner` running the `suite()` via `runner.run(suite())`
  d) a `if __name__ == "__main__":` calling `run()`
3. Add module to `runner.py` in subfolder by adding `test_xxxs`
  a) to import: in `try` add `import test_xxxs`, in `except` add `from test.subfolder import test_xxxs`
  b) to `test_modules` list

## Naming convention

- modules: `test_xxxs` (all lower case, ending with `s` since module can consist of multiple classes)
- class(es): `Xxxs` (first letter upper case, ending with `s` since class can consist of multiple test functions)
- functions: `test_yyy` (always starting with `test`since suite is build from all methods starting with `test`)
