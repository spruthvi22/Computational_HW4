"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Finding Fit using Bayesian inference using MCMC

"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import emcee
import corner


def log_likelihood(theta, x, y, yerr):  # Negative log likelyhood
    a, b, c = theta
    model = a*(x**2) + b*x + c
    sigma2 = yerr**2
    return 0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))


def log_prior(theta):    # Uniform Priors
    a, b, c = theta
    if -500.0 < a < 500 and -500 < b < 500 and -500 < c < 500:
        return 0.0
    return -np.inf


def log_probability(theta, x, y, yerr):     # Log Posterior
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp - log_likelihood(theta, x, y, yerr)


in_data = np.loadtxt('data.txt', skiprows=5, delimiter='&')
x = in_data[:, 1]
y = in_data[:, 2]
sigma_y = in_data[:, 3]
guess = (1.0, 1.0, 1.0)
soln = minimize(log_likelihood, guess, args=(x, y, sigma_y))
nwalkers, ndim = 50, 3
pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(50, 3, log_probability, args=(x, y, sigma_y))
sampler.run_mcmc(pos, 4000)
sample = sampler.get_chain()
plt.figure(1)
plt.title("Marcov Chains")
plt.subplot(3, 1, 1)
plt.plot(sample[:, :, 0], '-k', linewidth=0.2)  # a values
plt.ylabel("a")
plt.subplot(3, 1, 2)
plt.plot(sample[:, :, 1], '-k', linewidth=0.2)  # b values
plt.ylabel("b")
plt.subplot(3, 1, 3)
plt.plot(sample[:, :, 2], '-k', linewidth=0.2)  # c values
plt.ylabel("c")

# Combining Marcov chains for better accuracy, taking 500 burn in points
samples = np.zeros([175000, 3])
samples[:, 0] = sample[500:4000, :, 0].reshape((175000))
samples[:, 1] = sample[500:4000, :, 1].reshape((175000))
samples[:, 2] = sample[500:4000, :, 2].reshape((175000))
medians = np.median(samples, axis=0)
a_true, b_true, c_true = medians
# Plotting corner histogram with best fit and one-sigma values
fig = corner.corner(samples, labels=['a', 'b', 'c'], truths=[a_true, b_true, c_true], show_titles=True)

# Plotting input data points, best fit and possible solutions
plt.figure()
plt.errorbar(x, y, yerr=sigma_y, fmt='ok')
x_plt = np.linspace(np.min(x), np.max(x), 500)
y_plt = a_true*(x_plt**2) + b_true*x_plt + c_true
plt.xlabel("x")
plt.ylabel("y")
plt.title("Distribution and fit")
for i in np.random.randint(170000, size=100):
    y_plti = samples[i, 0]*(x_plt**2) + samples[i, 1]*x_plt + samples[i, 2]
    plt.plot(x_plt, y_plti, 'y', linewidth=0.1, alpha=0.4)

plt.plot(x_plt, y_plt, 'b', label="best fit")
plt.legend()
plt.show()
