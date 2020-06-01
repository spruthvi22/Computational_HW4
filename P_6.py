"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Finding randomly distributed numbers folowing given distribution using
rejection method

"""

import numpy as np
import matplotlib.pyplot as plt

# Creating x,y array of random numbers that follow exponential distribution
# Bounding the required distribution
n = 10000
x0 = np.random.rand(int(n))
x = -1 * np.log(x0)
y = np.random.rand(n) * 1.5 * np.exp(-1*x)
y_ac = []
x_ac = []              # Creating arrays of accepted and rejected values
y_rej = []
x_rej = []
for i in range(n):     
    if y[i] < np.sqrt(2/np.pi)*np.exp(-0.5*x[i]**2):
        y_ac.append(y[i])
        x_ac.append(x[i])
    else:
        y_rej.append(y[i])
        x_rej.append(x[i])


# Plotting the accepted and rejected points, histogram of distribution
plt.subplot(1, 2, 1, title="Distribution of points")
plt.plot(x_ac, y_ac, '.g', markersize=0.1)
plt.plot(x_rej, y_rej, '.r', markersize=0.1)
x_plt = np.linspace(0, 10, 100)
plt.plot(x_plt, 1.5 * np.exp(-1*x_plt), 'k', label="Bounding Distribution")
plt.plot(x_plt, np.sqrt(2/np.pi)*np.exp(-0.5*x_plt**2), 'b', label="Required distribution")
plt.xlabel("$x_i$")
plt.ylabel("$y_i$")
plt.legend()
plt.subplot(1, 2, 2, title="Gaussian Distribution")
plt.hist(x_ac, range=[0, 10])
plt.plot(x_plt, len(x_ac)*np.sqrt(2/np.pi)*np.exp(-0.5*x_plt**2), 'r')
plt.xlabel("$x_i$")
plt.ylabel("number in each bin")
plt.subplots_adjust(wspace=0.4)
plt.show()
