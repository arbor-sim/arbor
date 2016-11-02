#!/bin/bash
# @ job_name = miniapp
# @ partition = debug
## @ reservation = 
# @ initialdir = .
# @ output = miniapp_%j.out
# @ error = miniapp_%j.err
# @ total_tasks = 1
# @ cpus_per_task = 12
# @ node_usage = not_shared
# @ wall_clock_limit = 00:15:00

export OMP_NUM_THREADS=16
./run_instrumented.sh 

