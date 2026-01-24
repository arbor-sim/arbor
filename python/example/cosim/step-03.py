#!/usr/bin/env python3

from dataclasses import dataclass
from scipy.integrate import solve_ivp
import numpy as np


@dataclass
class Params:
    # coupling
    w_ee: float = 12
    w_ei: float = 10
    w_ie: float = 15
    w_ii: float = 0
    # input
    P: float = 1.25
    Q: float = 0
    # decay
    tau_e: float = 10  # ms
    tau_i: float = 10  # ms


def f(x: float, a: float = 1.3, m: float = 4.0):
    return 1.0 / (1.0 + np.exp(-a * (x - m)))


def wilson_cowan(t, y, ps: Params):
    E, I = y

    dE = (-E + f(ps.w_ee * E - ps.w_ei * I + ps.P)) / ps.tau_e
    dI = (-I + f(ps.w_ie * E - ps.w_ii * I + ps.Q)) / ps.tau_i

    return [dE, dI]


def step(t, dt, y, ps: Params):
    sol = solve_ivp(fun=wilson_cowan, t_span=[t, t + dt], y0=y, args=(ps,))
    return sol.y[:, 1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    y = np.array([0.2, 0.1])
    dt = 0.01
    ps = Params()

    t = 0
    ts = [t]
    Es = [y[0]]
    Is = [y[1]]
    while t < 1000:
        y = step(0, dt, y, ps)
        t += dt
        ts.append(t)
        Es.append(y[0])
        Is.append(y[1])

    fg, ax = plt.subplots()
    ax.plot(ts, Es)
    ax.plot(ts, Is)
    fg.savefig("step-03.png")
