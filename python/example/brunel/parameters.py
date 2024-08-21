#### define network parameters ####
order = 2500
NE = order * 4
NI = order
g = 8.0
eta = 1.5
epsilon = 0.1

J = 0.1
weight = J
delay = 1.5  # ms
theta = 20.0

CE = round(epsilon * NE)
CI = round(epsilon * NI)

#### define neuron parameter ####
tau_m = 20  # ms
V_th = theta  # mV
C_m = 250.0  # pF
E_L = 0.0  # mV
V_reset = 10.0  # mV
V_m = 0.0  # mV
t_ref = 2.0  # ms


##### for nest #####
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


#### define external input ####
nu_th = theta / (J * CE * tau_m)
nu_ex = eta * nu_th
rate = 1000.0 * nu_ex * CE  # Hz

#### simulation parameters ####
tfinal = 1000  # ms
dt = 1  # ms
seed = 42
