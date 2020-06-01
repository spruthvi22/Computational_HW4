"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Random numbers using MCMC

"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    if x > 3 and x < 7:
        return(1)
    else:
        return(0)


def LCG(n, seed=1):    # Uniform random numbers using LCG
    m = 2**31
    a = 1103515245
    c = 12345
    x = seed
    y = np.zeros(n)
    y[0] = (a*x + c) % m
    for i in range(n-1):
        y[i+1] = (a*y[i] + c) % m
    y = y/(m-1)
    return(y)


def Rand_gaus(n, sig=1):    # Gaussian distribution of desired sigma
    x1 = LCG(n)
    x2 = LCG(n)
    y1 = sig * np.sqrt(-2*np.log(x1)) * np.cos(2*np.pi*x2*(sig**2))
    return(y1)


nsteps = 10000
theta0 = 4.0
r = LCG(nsteps)
gaus = Rand_gaus(nsteps, sig=2)     # Taking gaussian as proposal pdf, with width that matches required pdf
theta = np.zeros(nsteps)
theta[0] = theta0

for i in range(nsteps-1):
    thet = theta[i] + gaus[i]
    if f(thet)/f(theta[i]) > r[i]:
        theta[i+1] = thet
    else:
        theta[i+1] = theta[i]


# Plotting Marcov chain and propability distribution
plt.subplot(2, 1, 1, title="Marcov Chain")
plt.plot(theta[0:100], '.r', markersize=0.5)
plt.xlabel("Iteration no:")
plt.ylabel("value")
plt.subplot(2, 1, 2, title="Density Histogram")
plt.hist(theta)
plt.xlabel("$x_i$")
plt.ylabel("number in each bin")
x = np.linspace(3, 7, 100)
y = nsteps/10*np.ones(len(x))
plt.plot(x, y, 'r')
plt.subplots_adjust(hspace=0.6)
plt.show()
