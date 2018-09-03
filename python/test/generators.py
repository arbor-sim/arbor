import pyarb as arb

sr = arb.regular_schedule()
sr.tstart = 10
sr.tstop = 20
sr.dt = 1

se = arb.explicit_schedule()
se.times = [1, 2, 3, 4.5]

gr = arb.event_generator(10, 0.2, sr)
ge = arb.event_generator(42, -0.2, se)
