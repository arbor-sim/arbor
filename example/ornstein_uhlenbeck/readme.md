## Miniapp for demonstrating stochastic processes

The simulation consists of a single cell with a stochastic process (Ornstein-Uhlenbeck) painted on
its control volumes. The stochastic process is described by a linear mean-reverting stochastic
differential equation which is specified in the accompanying NMODL file. All processes start from
the same initial condition and are averaged over the control volumes at each time step to generate an
ensemble statistic. These results are then compared to the analytical solution.
