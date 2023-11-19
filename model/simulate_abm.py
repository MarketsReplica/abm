import argparse
import pandas as pd
import numpy as np
from scipy.io import loadmat

def randpl(n, alpha, N):
    # Generate n observations distributed as power law
    x = (1 - np.random.rand(n))**(-1/(alpha - 1))
    x = np.round(x / np.sum(x) * N)
    x[x < 1] = 1

    dx = np.sum(x) - N
    while dx != 0:
        if dx < 0:
            # If dx is negative, add 1 to randomly selected elements
            id = np.random.choice(np.where(x > 1)[0], size=abs(dx), replace=False)
            x[id] = x[id] + 1
        elif dx > 0:
            # If dx is positive, subtract 1 from randomly selected elements
            id = np.where(x > 1)[0]
            id = np.random.choice(id, size=min(abs(dx), len(id)), replace=False)
            x[id] = x[id] - 1

        dx = np.sum(x) - N

    return x.astype(int)

T = 12
G = 62

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, default=2019)
parser.add_argument('-q', '--quarter', type=int, choices=range(1, 5), default=1)
kw = parser.parse_args()
parameters = loadmat(f"model/parameters/{kw.year}Q{kw.quarter}.mat")
initial_conditions = loadmat(f"model/initial_conditions/{kw.year}Q{kw.quarter}.mat")

kw = kw.__dict__.copy()
kw.update(parameters)
kw.update(initial_conditions)

print(parameters)

for k, v in parameters.items():  # unpack from .m
    if hasattr(v, 'shape'):
        v = v.reshape(-1)
        if len(v) == 1:
            v = v[0]
        kw[k] = v

I = kw['I_s'].sum()
G_i = np.zeros(I, dtype=int)
for g in range(0, kw['G']):
    i = int(kw['I_s'][0: g].sum())
    G_i[i + 1: i + kw['I_s'][g]] = g


alpha_bar_i = kw['alpha_s'][G_i]
beta_i = kw['beta_s'][G_i]
kappa_i = kw['kappa_s'][G_i]
delta_i = kw['delta_s'][G_i]
w_bar_i = kw['w_s'][G_i]
tau_Y_i = kw['tau_Y_s'][G_i]
tau_K_i = kw['tau_K_s'][G_i]

kw['Y'] = (kw['Y'], np.zeros(T))
kw['pi'] = (kw['pi'], np.zeros(T))

P_bar = P_bar_HH = P_bar_CF = 1
P_bar_g = np.ones(G)

N_i = np.zeros(I)
for g in range(0, kw['G']):
    indices = np.where(G_i == g)[0]
    N_i[indices] = randpl(len(indices), 2, kw['N_s'][g - 1])

Y_i = alpha_bar_i * N_i
Q_d_i = Y_i
P_i = np.ones(I)
S_I = np.zeros(I)
K_i = Y_i / (kw['omega'] * kappa_i)
M_i=Y_i / (kw['omega'] * beta_i)
L_i=kw['L_i'] * K_i / sum(K_i)


# Average operating margin per firm (but sectoral)
pi_bar_i= 1 - (1 + kw['tau_SIF']) * w_bar_i / alpha_bar_i-delta_i / kappa_i-1 / beta_i-tau_K_i-tau_Y_i
D_i = kw['D_I'] * max(0, pi_bar_i *Y_i) / sum(max(0, pi_bar_i *Y_i))

# Interest rate charged by banks is CB rate + a risk premium
r=kw['r_bar'] + kw['mu']

# Profit by firm is operating margin per unit sold * units sold - interst
# on loans + interest on deposits at the "Rothschild" bank
Pi_i = pi_bar_i * Y_i-r * L_i + kw['r_bar'] * max(0, D_i)

# Rothschild bank profit
Pi_k=kw['mu'] * sum(L_i)+kw['r_bar'] * kw['E_k']

# See appendix E.2 p. 29 ff
# Attribute average sectoral wage to households for sectors/firms they're
# employed at, first initialize
H_W = kw['H_act'] - I - 1
w_h = np.zeros(H_W)
O_h = np.zeros(H_W)
V_i = N_i.copy()
h = 0  # Initialize h to 0 in Python (Python indices start from 0)

# These people work and get average sectoral wages
for i in range(I):
    while V_i[i] > 0:
        O_h[h] = i
        w_h[h] = w_bar_i[i]
        V_i[i] = V_i[i] - 1
        h = h + 1

# These people receive unemployment benefits as a fraction of their
# hypothetical previous sectoral wage.
w_h[O_h == 0] = kw['w_UB'] / kw['theta_UB']


H = kw['H_act'] + kw['H_inact']
Y_h = np.zeros(H)

# Here, calculate disposable income by HH, i.e. average wage minus taxes +
# social benefits, or capital income
for h in range(1, H+1):
    # Employed and unemployed households
    if h <= H_W:
        if O_h[h-1] != 0:
            Y_h[h-1] = (w_h[h-1] * (1 - kw['tau_SIW'] - kw['tau_INC'] * (1 - kw['tau_SIW'])) + kw['sb_other']) * P_bar_HH
        else:
            Y_h[h-1] = (kw['theta_UB'] * w_h[h-1] + kw['sb_other']) * P_bar_HH
    # Inactive HHs receive social benefits
    elif H_W < h <= H_W + kw['H_inact']:
        Y_h[h-1] = (kw['sb_inact'] + kw['sb_other']) * P_bar_HH
    # Capitalists receive dividend income
    elif H_W + kw['H_inact'] < h <= H_W + kw['H_inact'] + I:
        i = h - (H_W + kw['H_inact'])
        Y_h[h-1] = kw['theta_DIV'] * (1 - kw['tau_INC']) * (1 - kw['tau_FIRM']) * max(0, Pi_i[i-1]) + kw['sb_other'] * P_bar_HH
    # "Rothschilds" receive financial profit dividend income
    elif H_W + kw['H_inact'] + I < h <= H:
        Y_h[h-1] = kw['theta_DIV'] * (1 - kw['tau_INC']) * (1 - kw['tau_FIRM']) * max(0, Pi_k) + kw['sb_other'] * P_bar_HH



# Households get attributed initial deposits according to their average
# income as a proxy for their wealth
D_h = kw['D_H'] * Y_h / np.sum(Y_h)
K_h = kw['K_H'] * Y_h / np.sum(Y_h)

# Initial advances from the central bank (Dk(0)), as in equation (A.68) in the main text.
D_k = np.sum(D_i) + np.sum(D_h) + kw['E_k'] - np.sum(L_i)


T, G, T_prime = map(kw.__getitem__, ('T', 'G', 'T_prime'))

nominal_gdp = np.zeros(T+1)
real_gdp = np.zeros(T+1)
nominal_gva = np.zeros(T+1)
real_gva = np.zeros(T+1)
nominal_household_consumption = np.zeros(T+1)
real_household_consumption = np.zeros(T+1)
nominal_government_consumption = np.zeros(T+1)
real_government_consumption = np.zeros(T+1)
nominal_capitalformation = np.zeros(T+1)
real_capitalformation = np.zeros(T+1)
nominal_fixed_capitalformation = np.zeros(T+1)
real_fixed_capitalformation = np.zeros(T+1)
nominal_fixed_capitalformation_dwellings = np.zeros(T+1)
real_fixed_capitalformation_dwellings = np.zeros(T+1)
nominal_exports = np.zeros(T+1)
real_exports = np.zeros(T+1)
nominal_imports = np.zeros(T+1)
real_imports = np.zeros(T+1)
operating_surplus = np.zeros(T+1)
compensation_employees = np.zeros(T+1)
wages = np.zeros(T+1)
taxes_production = np.zeros(T+1)
nominal_sector_gva = np.zeros((T+1, G))
real_sector_gva = np.zeros((T+1, G))
euribor = np.zeros(T+1)
gdp_deflator_growth_ea = np.zeros(T+1)
real_gdp_ea = np.zeros(T+1)

# Initialize the vectors of the main model output with t=1 initial period
nominal_gdp[0] = np.sum(Y_i * (1 - 1 / beta_i)) + np.sum(Y_h) * kw['psi'] / (1 / kw['tau_VAT'] + 1) + kw['tau_G'] * kw['C_G'][T_prime] + np.sum(Y_h) * kw['psi_H'] / (1 / kw['tau_CF'] + 1) + kw['tau_EXPORT'] * kw['C_E'][T_prime]
real_gdp[0] = nominal_gdp[0]
nominal_gva[0] = np.sum(Y_i * ((1 - tau_Y_i) - 1 / beta_i))
real_gva[0] = nominal_gva[0]
nominal_household_consumption[0] = np.sum(Y_h) * kw['psi']
real_household_consumption[0] = nominal_household_consumption[0]
nominal_government_consumption[0] = (1 + kw['tau_G']) * kw['C_G'][T_prime]
real_government_consumption[0] = nominal_government_consumption[0]
nominal_capitalformation[0] = np.sum(Y_i * delta_i / kappa_i) + np.sum(Y_h) * kw['psi_H']
real_capitalformation[0] = nominal_capitalformation[0]
nominal_fixed_capitalformation[0] = nominal_capitalformation[0]
real_fixed_capitalformation[0] = nominal_capitalformation[0]
nominal_fixed_capitalformation_dwellings[0] = np.sum(Y_h) * kw['psi_H']
real_fixed_capitalformation_dwellings[0] = nominal_fixed_capitalformation_dwellings[0]
nominal_exports[0] = (1 + kw['tau_EXPORT']) * kw['C_E'][T_prime]
real_exports[0] = nominal_exports[0]
nominal_imports[0] = kw['Y_I'][T_prime]
real_imports[0] = nominal_imports[0]
operating_surplus[0] = np.sum(Y_i * (1 - ((1 + kw['tau_SIF']) * w_bar_i / alpha_bar_i + 1 / beta_i)) - tau_K_i * Y_i - tau_Y_i * Y_i)
compensation_employees[0] = (1 + kw['tau_SIF']) * np.sum(w_bar_i * N_i)
wages[0] = np.sum(w_bar_i * N_i)
taxes_production[0] = np.sum(tau_K_i * Y_i)


for g in range(1, G+1):
    nominal_sector_gva[0, g-1] = np.sum(Y_i[G_i == g] * ((1 - tau_Y_i[G_i == g]) - 1 / beta_i[G_i == g]))

real_sector_gva[0, :] = nominal_sector_gva[0, :]
euribor[0] = kw['r_bar']
gdp_deflator_growth_ea[0] = kw['pi_EA']
real_gdp_ea[0] = kw['Y_EA']


parameters = pd.DataFrame()