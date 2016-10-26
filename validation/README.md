# Validation data and generation

## Sub-directory organization

`validation/data`
 ~ Generated validation data

`validation/ref`
 ~ Reference models

`validation/ref/neuron`
 ~ NEURON-based reference models, run with `nrniv -python`

`validation/ref/numeric`
 ~ Direct numerical and analytic models

## Data generation

Data is generated via the `validation_data` CMake target, which is
a prerequisite for the `validation.exe` test executable.


