# define network parameters
order = 2500
NE = order * 4
NI = order
g = 8.0
eta = 1.5
epsilon = 0.1

J = 0.1
weight = J
delay = 1.5
theta = 20.0

CE = round(epsilon * NE)
CI = round(epsilon * NI)

# define neuron parameter
tau_m = 20
V_th = theta
C_m = 250.0
E_L = 0.0
V_reset = 10.0
V_m = 0.0
t_ref = 2.0


# for nest
neuron_model = "iaf_psc_delta"

neuron_params = {
    "C_m": C_m,
    "tau_m": tau_m,
    "t_ref": t_ref,
    "E_L": E_L,
    "V_reset": V_reset,
    "V_m": V_m,
    "V_th": theta,
}


# define external input
nu_th = theta / (J * CE * tau_m)
nu_ex = eta * nu_th
rate = 1000.0 * nu_ex * CE

# simulation parameters ####
tfinal = 1000
dt = 1
seed = 42
