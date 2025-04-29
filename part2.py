import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import part1_fun
import part2_fun

# [1] Credit spread, corporate bond yield and zero
nss_coeff = [.0338239271393464, .0106444272253698, .0334588125051062e-05, .043114875282879, 0.767301405556937, 19.3521664323374]
aa_yield = [4.50, 4.53, 4.59, 4.66, 4.73, 4.81, 5.01, 5.28, 5.42, 5.55, 5.67, 5.77]
data = pd.DataFrame(
    aa_yield,
    index=[0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 30],
    columns=["AA_yield"]
)
data = data / 100

# t-bill zcy
rf = part1_fun.nss(data.index, *nss_coeff)
tscs = data.AA_yield - rf

fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()
ax1.plot((tscs) * 10000, color = "black", marker = "D", markersize = 4, linestyle = "-", label="TSCS")
ax1.set_xlabel("Years")
ax1.set_ylabel("Credit Spread (bps)")
ax2.plot(data.index, data.AA_yield * 100, color = "red", marker = "D", markersize = 4,linestyle = "-.", label="AA Bond Yield")
ax2.plot(data.index, rf * 100, color = "gray", marker = "D", markersize = 4, linestyle = "-.", label="Risk Free rate")
ax2.set_ylabel("Yield (%)")
ax1.legend(loc=[0.015,0.82],facecolor='none', edgecolor='none')
ax1.set_title("Term structure of credit spread & ZCY")
plt.legend(facecolor='none', edgecolor='none')
plt.gcf()
plt.savefig("plot/tscs_part2.pdf")
plt.show()

# Table for data
latex_data = pd.DataFrame([data.index, data.AA_yield*100, rf*100,tscs*10000]).T
latex_data.columns = ["Maturity", "AA Yield (%)", "NSS Yield (%)", "TSCS (bps)"]
latex_data = latex_data.round(4)



# [3] Calibration
# [3.1] Calibrate the r_t to nss_zcy
T_grid = np.array(data.index)
def sse(para):
    yld_theory = part2_fun.yld(0, T_grid, *para)
    error = sum((yld_theory - rf)**2)
    print(error)
    return error


# kappa, theta, sigma, x0
bunds = [(0,0.5),(0,0.1),(0,0.5),(0,0.1)]
opt_result = minimize(sse,
                      x0= np.repeat(0.05,4),
                      method='COBYLA',
                      bounds=bunds,
                      options={'maxiter': 1000000}
                      )
opt_rf = opt_result["x"]
# [0.00941375, 0.09876645, 0.00101344, 0.03980174]
yld_clib_rf = part2_fun.yld(0, T_grid, *opt_rf)
yld_clib_rf = part2_fun.yld(0, np.linspace(0.5,30,200), *opt_rf)


# [3.2] Calibrate the yield to AA bond yield
# conditioning on the calibrated parameters in the previous plot
def sse(para):
    yld_theory = part2_fun.yld_AA(0, T_grid, *para, *opt_rf)
    error = sum((yld_theory - data.AA_yield)**2)
    print(error)
    return error

bunds = [(0,0.5),(0,0.15),(0,0.5),(0,0.1),(-0.5, 0.5),(-0.5, 0.5)]
opt_result = minimize(sse,
                      x0= np.repeat(0.05,6),
                      method='Nelder-Mead',
                      bounds=bunds,
                      options={'maxiter': 1000000}
                      )
pack_BM = opt_result["x"]
# [0.00941375, 0.09876645, 0.00101344, 0.03980174]
yld_clib_AA = part2_fun.yld_AA(0, T_grid, *pack_BM, *opt_rf)
yld_clib_AA = part2_fun.yld_AA(0,  np.linspace(0.5,30,200), *pack_BM, *opt_rf)

# Plt
fig, ax = plt.subplots(figsize=(8, 6))
ax1 = ax.twinx()
ax.plot(np.linspace(0.5,30,200),yld_clib_AA * 100, color = "black", linestyle = "--", label="AA Yield")
ax.scatter(data.index,data.AA_yield * 100, color = "red", marker = "D", s = 12, label="Observed AA Yield")

ax.plot(np.linspace(0.5,30,200), yld_clib_rf * 100, color = "grey", linestyle = "-.", label="NSS ZCB Yield")
ax.scatter(data.index,rf * 100, color = "darkorange", marker = "D", s = 12, label="Observed ZCB Yield")

ax1.plot(np.linspace(0.5,30,200), (yld_clib_AA - yld_clib_rf) * 10000, label = "TSCS")
ax1.set_ylabel("TSCS (bps)")
ax1.legend(loc=[0.013,0.75],facecolor='none', edgecolor='none')

ax.set_xlabel("Years")
ax.set_ylabel("Yield (%)")
ax.legend(loc=[0.013,0.8],facecolor='none', edgecolor='none')
ax.set_title("Calibration v.s. observed yield")
# plt.legend(facecolor='none', edgecolor='none')
plt.gcf()
plt.savefig("plot/tscs_part31.pdf")
plt.show()





