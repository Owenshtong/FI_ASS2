import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.optimize import minimize


# CIR ZCB price
def a_cir(t, T, kappa, theta, sigma):
    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    numerator = 2 * gamma * np.exp((kappa + gamma)/2 * (T-t))
    denominator = (kappa + gamma) * (np.exp(gamma * (T-t)) - 1) + 2 * gamma
    return (numerator / denominator) ** (2 * kappa * theta / sigma**2)

def b_cir(t, T, kappa, sigma):
    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    numerator = 2 * (np.exp(gamma * (T-t))-1)
    denominator = (kappa + gamma) * (np.exp(gamma * (T-t)) - 1) + 2 * gamma
    return numerator / denominator

def P_cir(t, T, kappa, theta, sigma, x0):
    a = a_cir(t, T, kappa, theta, sigma)
    b = b_cir(t, T, kappa, sigma)
    return a * np.exp(-1 * b * x0)

def yld(t, T, kappa, theta, sigma, x0):
    P = P_cir(t, T, kappa, theta, sigma, x0)
    return -1/(T-t) * np.log(P)

def yld_AA(t, T, kappa_s, theta_s, sigma_s, s0, alpha, beta, kappa_r, theta_r, sigma_r, x0):
    P_y = P_cir(t,T, kappa_r, theta_r * (1 + beta), sigma_r * np.sqrt(beta + 1), x0*(1 + beta))
    P_s = P_cir(t, T, kappa_s, theta_s, sigma_s, s0)
    return -1/(T-t) * (np.log(P_y) + np.log(P_s) - alpha * T)


# Explicit finite difference
def mu_x(kappa, theta, xt):
    return kappa * (theta - xt)

def vol_x(sigma, xt):
    return sigma * np.sqrt(xt)

# Coeff
# def B(dt, dr, vol_r, mu_r):
#     return (dt * vol_r) / (2 * dr**2) - dt * mu_r/(2 * dr)
#
# def D(dt, ds, vol_s, mu_s):
#     return (dt * vol_s) / (2 * ds**2) - dt * mu_s/(2 * ds)
#
# def E(dt, dr, ds, ind_r, ind_s, vol_r, vol_s):
#     r_ij = ind_r * dr + ind_s * ds
#     return 1 - (dt * vol_r) / dr**2 - (dt * vol_s) / ds**2 - r_ij * dt
#
# # def E(dt,dr,ds, r, s , vol_r, vol_s, alpha, beta):
# #     r_ij = alpha + (1+beta) * r + s
# #     return 1 - (dt * vol_r) / dr**2 - (dt * vol_s) / ds**2 - r_ij * dt
#
# def H(dt, dr,vol_r, mu_r):
#     return (dt * vol_r) / (2 * dr**2) + dt * mu_r/(2 * dr)
#
# def F(dt, ds, vol_s, mu_s):
#     return (dt * vol_s) / (2 * ds**2) + dt * mu_s/(2 * ds)
#


def B(dt, dr, vol_r, mu_r):
    return (dt * vol_r**2) / (2 * dr**2) - dt * mu_r/(2 * dr)

def D(dt, ds, vol_s, mu_s):
    return (dt * vol_s**2) / (2 * ds**2) - dt * mu_s/(2 * ds)
#
# def E(dt, dr, ds, ind_r, ind_s, vol_r, vol_s):
#     r_ij = ind_r * dr + ind_s * ds
#     return 1 - (dt * vol_r**2) / dr**2 - (dt * vol_s**2) / ds**2 - r_ij * dt

def E(dt,dr,ds, r, s , vol_r, vol_s, alpha, beta):
    r_ij = alpha + (1+beta) * r + s
    return 1 - (dt * vol_r**2) / dr**2 - (dt * vol_s**2) / ds**2 - r_ij * dt

def H(dt, dr,vol_r, mu_r):
    return (dt * vol_r**2) / (2 * dr**2) + dt * mu_r/(2 * dr)

def F(dt, ds, vol_s, mu_s):
    return (dt * vol_s**2) / (2 * ds**2) + dt * mu_s/(2 * ds)


