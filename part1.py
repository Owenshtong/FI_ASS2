### Perform the KMV estimation on extended Merton's model ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import part1_fun
from firm_cls import Firm
from scipy.stats import norm

# Risk-free rate
rf = pd.read_csv("rf.csv", index_col=0, parse_dates=True)/100
rf = rf.interpolate(method = "time")

# Quote the firm data: Black gold
tic = "GOLD"
start = "2023-12-31"
end = "2024-12-31"
pernmo = "71298"
gvkey = "002055"


GOLD = Firm(
    tic,
    start,
    end,
    pernmo,
    gvkey,
    rf
)

# Interpolated long and short-term debt, Market cap
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(GOLD.data.K / 1e9,color = "black", linestyle = "-", label = "$L_t$")
ax.set_ylabel(r"$L_t (10^9)$")
ax.set_title("Merton's Debt Level over year 2024")
ax2 = ax.twinx()
ax2.plot(GOLD.data.Mrk_cap / 1e10,color = "red", linestyle = "-.",label = "Market cap")
ax2.set_ylabel(r"Market Cap $(10^{10})$")
ax.legend(loc=[0.015,0.82],facecolor='none', edgecolor='none')
ax2.legend(loc = [0.015,0.9], facecolor='none', edgecolor='none')
plt.gcf()
plt.savefig("plot/Kt.pdf")
plt.show()

# KMV
sigma, v0 = part1_fun.KMV(0, GOLD.data.Mrk_cap, GOLD.data.K, GOLD.data.rf)




## Interpolate frontier by linearly interpolate
# Observed yield
nss_coef = [.013997809781669e-05, .0417537838746407, -1.243277675590, 1.34379235533314, 12.3652373905849, 12.9049079240786]
t_grid = np.linspace(1, 20, 500)
y_obs = pd.DataFrame(part1_fun.nss(t_grid, *nss_coef), index=t_grid, columns=["zcy"])

# Interpolate the debt structure
debt_1 = GOLD.data.K.iloc[-1]
debt_20 = GOLD.data.ld.iloc[-1]
debt = pd.DataFrame({
    "debt": [debt_1, debt_20]
}, index = [1,20])


data = pd.concat([y_obs, debt],axis=1).sort_index()
data = data.interpolate(method = "index")





# Get the yield
D_mod = part1_fun.D(v0, sigma, 0, data.debt, data.zcy, data.index)
y_mod = -1/data.index * np.log(D_mod/data.debt)
tscs_nss = y_mod - data.zcy

fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()
ax1.plot((tscs_nss) * 10000, color = "black", linestyle = "-", label="TSCS")
ax1.set_xlabel("Years")
ax1.set_ylabel("Credit Spread (bps)")
ax2.plot(y_mod * 100, color = "red", linestyle = "-.", label="Debt yield")
ax2.plot(data.zcy * 100, color = "gray", linestyle = "-.", label="T-Bond ZCY")
ax2.set_ylabel("Yield (%)")
ax1.legend(loc=[0.015,0.82],facecolor='none', edgecolor='none')
ax1.set_title("Term structure of credit spread & ZCY")
plt.legend(facecolor='none', edgecolor='none')
plt.gcf()
plt.savefig("plot/tscs.pdf")
plt.show()



