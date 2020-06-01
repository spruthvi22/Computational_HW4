"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Finding area using monte-carlo method

"""

import numpy as np
import matplotlib.pyplot as plt


# Solving using rejection mponte carlo method
n2 = 10000
y2 = (2*np.random.rand(n2, 2))-1
y2_ac = []
y2_rej = []
for i in range(n2):
    if (y2[i, 0]**2+y2[i, 1]**2) < 1:
        y2_ac.append(y2[i, :])
    else:
        y2_rej.append(y2[i, :])
y2_ac = np.array(y2_ac)
y2_rej = np.array(y2_rej)

plt.plot(y2_ac[:, 0], y2_ac[:, 1], '.g', markersize=0.2)
plt.plot(y2_rej[:, 0], y2_rej[:, 1], '.r', markersize=0.2)
x_plt = np.linspace(-1, 1, 100)
plt.plot(x_plt, np.sqrt(1-x_plt**2), 'k')
plt.plot(x_plt, -1*np.sqrt(1-x_plt**2), 'k')
plt.xlabel("$x_i$")
plt.ylabel("$y_i$")
plt.title("Accepted and Rejected points in Rejection Method")
vol2_re = (2**2)*(np.shape(y2_ac)[0]/np.shape(y2)[0])
print("Integral 2d Using Rejection Method = ", vol2_re)


def f(y):
    k = np.zeros(np.shape(y)[0])
    for i in range(np.shape(y)[0]):
        if np.sum(y[i, :]**2) < 1:
            k[i] = 1
    return(k)


# Using mean value monte carlo method
n2 = 10000
y2 = (2*np.random.rand(n2, 2))-1
fy2 = f(y2)
vol2_me = (2**2)*np.mean(fy2)
print("Integral 2d Using Mean Value Method = ", vol2_me)

n10 = 10000
y10 = (2*np.random.rand(n10, 10))-1
fy10 = f(y10)
vol10_me = (2**10)*np.mean(fy10)
print("Integral 10d Using Mean Value Method = ", vol10_me)
plt.show()
