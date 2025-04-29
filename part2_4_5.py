import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import part1_fun
import part2_fun

# Duffee(1999) paramters
r_pra = [0.04382889, 0.05719049, 0.01938527, 0.03948566]
s_pra = [0.0414006 ,  0.06976397,  0.06907816,  0.03086196, -0.02982357,
        0.08256708] # including alpha and beta
kappa_r, theta_r, sigma_r, r0 = r_pra
kappa_s, theta_s, sigma_s, s0, alpha, beta = s_pra

###### Build the underlying asset scheme #####

# Hyper param setup
r_max = 0.2
s_max = 0.2
N_r = 100
N_s = 100
dr = r_max/N_r
ds = s_max/N_s
N_t = 101
T = 5
dt = T/N_t
M = 1000000
K_t = M # Call at par
c_per_real = .02
c_dollar = M * c_per_real


# Generate the 3d array, store using time in t_grid as key
t_grid = np.round(np.linspace(0,T,N_t),4)
r_grid = np.linspace(0,r_max,N_r)
s_grid = np.linspace(0,s_max,N_s)

# Compute g_c values
l = []
bound = 4.5
for i in list(range(N_t-1, -1, -1)):
        print(i)
        t = t_grid[i]
        if t == T:
                sheet_T = np.reshape(np.repeat(M + c_dollar, N_s * N_r), (N_s, N_r))
                l.append(sheet_T)
        else:
                # new vault
                sheet_next = l[-1]
                sheet_t = np.zeros((N_s, N_r))

                # Fill up the inner points
                for y in range(1, N_r - 2):
                        st = s_grid[y]
                        for x in range(1, N_s - 2):
                                rt = r_grid[x]

                                # get the coefficients
                                mu_r = part2_fun.mu_x(kappa_r, theta_r, rt)
                                vol_r = part2_fun.vol_x(sigma_r, rt)
                                mu_s = part2_fun.mu_x(kappa_s, theta_s, st)
                                vol_s = part2_fun.vol_x(sigma_s, st)

                                B = part2_fun.B(dt, dr, vol_r, mu_r)
                                D = part2_fun.D(dt, ds, vol_s, mu_s)
                                E = part2_fun.E(dt, dr, ds, x, y, vol_r, vol_s)
                                H = part2_fun.H(dt, dr, vol_r, mu_r)
                                F = part2_fun.F(dt, ds, vol_s, mu_s)

                                # continuous value
                                cont = B * sheet_next[x-1, y] + D * sheet_next[x,y-1] + \
                                       E * sheet_next[x,y] + F * sheet_next[x,y+1] + H * sheet_next[x+1, y]

                                # assign value to the new sheet
                                sheet_t[x,y] = min(K_t, cont)

                # Fill up the boundary
                # Right and Left boundary
                for y in range(1,N_r - 2):
                        sheet_t[N_s-1, y] = 2 * sheet_t[N_s-2, y] - sheet_t[N_s-3, y] # linear deep ITM
                        sheet_t[0, y] = sheet_t[1, y] # Flat in r
                # Top and bottom boundary
                for x in range(N_r):
                        sheet_t[x, N_r-1] = 2 * sheet_t[x, N_s-2] - sheet_t[x, N_s-3] # linear deep ITM
                        sheet_t[x, 0] = sheet_t[x, 1] # Flat in s
                l.append(sheet_t)























