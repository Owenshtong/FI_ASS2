import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import part1_fun
import part2_fun
from scipy.interpolate import griddata
from scipy.optimize import fsolve

# Duffee(1999) paramters
r_pra = [0.04382889, 0.05719049, 0.01938527, 0.03948566]
s_pra = [0.0414006 ,  0.06976397,  0.06907816,  0.03086196, -0.02982357,
        0.08256708] # including alpha and beta

kappa_r, theta_r, sigma_r, r0 = r_pra
kappa_s, theta_s, sigma_s, s0, alpha, beta = s_pra


###### Build the underlying asset scheme #####

# Hyper param setup
r_max = 0.5
s_max = 0.5
N_r = 100
N_s = 100
dr = r_max/N_r
ds = s_max/N_s
N_t = 2000
T = 5
dt = T/N_t
M = 1000000
K_t = M # Call at par
c_per_real = .02
c_dollar = M * c_per_real


# Generate the 3d array, store using time in t_grid as the key
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
                for y in range(1, N_r - 1):
                        st = s_grid[y]
                        for x in range(1, N_s - 1):
                                rt = r_grid[x]

                                # get the coefficients
                                mu_r = part2_fun.mu_x(kappa_r, theta_r, rt)
                                vol_r = part2_fun.vol_x(sigma_r, rt)
                                mu_s = part2_fun.mu_x(kappa_s, theta_s, st)
                                vol_s = part2_fun.vol_x(sigma_s, st)

                                B = part2_fun.B(dt, dr, vol_r, mu_r)
                                D = part2_fun.D(dt, ds, vol_s, mu_s)
                                E = part2_fun.E(dt, dr, ds, rt, st, vol_r, vol_s, alpha, beta)
                                H = part2_fun.H(dt, dr, vol_r, mu_r)
                                F = part2_fun.F(dt, ds, vol_s, mu_s)


                                # continuous value
                                cont = B * sheet_next[x-1, y] + D * sheet_next[x,y-1] + \
                                       E * sheet_next[x,y] + F * sheet_next[x,y+1] + H * sheet_next[x+1, y]

                                # check coupon payment
                                if t <= bound:
                                        cont = cont + c_dollar

                                # assign value to the new sheet
                                sheet_t[x,y] = min(K_t, cont)

                # Fill up the boundary
                # Right and Left boundary
                for y in range(1,N_r - 1):
                        sheet_t[N_s-1, y] = 2 * sheet_t[N_s-2, y] - sheet_t[N_s-3, y] # linear deep ITM
                        sheet_t[0, y] = sheet_t[1, y] # Flat in r
                # Top and bottom boundary
                for x in range(N_r):
                        sheet_t[x, N_r-1] = 2 * sheet_t[x, N_s-2] - sheet_t[x, N_s-3] # linear deep ITM
                        sheet_t[x, 0] = sheet_t[x, 1] # Flat in s
                if t <= bound:
                        bound = bound - 0.5
                l.append(sheet_t)


t0 = l[-1]

# Bond without a callable option
D_non_callable = 0
for t in np.linspace(0.5,5,10):
        if t == 5:
                cf = M +  c_dollar
        else:
                cf = c_dollar

        D_non_callable += cf * part2_fun.P_cir(0, t, kappa_r, theta_r * (1 + beta), sigma_r * np.sqrt(1 + beta), r0) * part2_fun.P_cir(0, t, kappa_s, theta_s, sigma_s, s0) * np.exp(-alpha * t)


# Interpolate
r_cord, s_cord = np.meshgrid(r_grid, s_grid)

D_callable = griddata((r_cord.ravel(), s_cord.ravel()), t0.ravel(), (r0,s0), method='linear')

# Cord plot
fig = plt.figure()
fig.set_size_inches(10,6)
cf = plt.contourf(r_cord, s_cord, t0, levels=300, cmap='binary')
plt.colorbar(cf, label='Debt Value', format='%.1e')
plt.scatter(r0, s0, color="red",label = "current point")
plt.title("Contour plot of debt value at time 0")
plt.xlabel(r"$r_0$")
plt.ylabel(r"$s_0$")
plt.legend()
plt.gcf()
plt.savefig("plot/court.pdf")
plt.show()

# Compute the yield

def D_dcf(y, D):
        coupon_t = np.linspace(0.5, 5, 10)
        acc = 0
        for t in coupon_t:
               acc += c_dollar * np.exp(- t * y)
        acc += M * np.exp(-5 * y)
        return D - acc

y_callable = fsolve(D_dcf, 0.01, args=(D_callable))
y_non_callable = fsolve(D_dcf, 0.01, args=(D_non_callable))
