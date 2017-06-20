A json record for a meter measurement is a json object.

Each Object corresponds to a derived measurement:
  * `name`: a string describing the measurement
  * `units`: a string with SI units for measurements
  * `measurements`: a json Array of measurements, with one entry for the
    each checkpoint. The first enry is the measure of resources consumed
    between the call to `meter_manager::start()` and the first checkpoint, the
    second entry measure between the first and second checkpoints, and son on.
  * each measurement is itself a numeric array, with one recording for each
    domain in the global communicator

For example, the output of a meter for measuring wall time where 4 checkpoints
were taken on 4 MPI ranks could be represented as follows:

```json
  {
    "name": "walltime",
    "units": "s",
    "measurements": [
      [ 0.001265837, 0.001344004, 0.001299362, 0.001195762, ],
      [ 0.014114013, 0.015045662, 0.015071675, 0.014209514, ],
      [ 1.491986631, 1.491121134, 1.490957219, 1.492064233, ],
      [ 0.00565307, 0.004375347, 0.002228206, 0.002483978, ]
    ]
  }
```
