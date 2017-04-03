A json record for a meter measurement is a json object.

Each Object corresponds to a derived measurement:
  * `name`: a string describing the measurement
  * `units`: a string with SI units for measurements
  * `measurements`: a json Array of measurements, with one
    entry per checkpoint (corresponding to a call to
    meter::take_reading)
  * each measurement is itself a numeric array, with one
    recording for each domain in the global communicator

For example, the output of a meter for measuring wall time where 5 readings
were taken on 4 MPI ranks could be represented as follows:

```json
  {
    "name": "walltime",
    "units": "s",
    "measurements": [
      [ 0, 0, 0, 0, ],
      [ 0.001265837, 0.001344004, 0.001299362, 0.001195762, ],
      [ 0.014114013, 0.015045662, 0.015071675, 0.014209514, ],
      [ 1.491986631, 1.491121134, 1.490957219, 1.492064233, ],
      [ 0.00565307, 0.004375347, 0.002228206, 0.002483978, ]
    ]
  }
```
