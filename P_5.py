"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Finding Gaussian distributed random numbers using Box-Muller

"""

import numpy as np
import matplotlib.pyplot as plt


n = 10000
x1 = np.random.rand(n)      # Two unifrom distributions
x2 = np.random.rand(n)
# Gaussian distribution using box muller method
y1 = np.sqrt(-2*np.log(x1)) * np.cos(2*np.pi*x2)
y2 = np.sqrt(-2*np.log(x1)) * np.sin(2*np.pi*x2)

# Plotting the Uniform and Gaussian distributions
plt.subplot(1, 2, 1, title="Uniform distribution")
plt.hist(x1)
x = np.linspace(0, 1, 100)
y = n/10*np.ones(len(x))
plt.plot(x, y, 'r')
plt.xlabel("$x_i$")
plt.ylabel("number in each bin")
plt.subplot(1, 2, 2, title="Gaussian Distribution")
plt.hist(y1, range=[-5, 5])
x = np.linspace(-5, 5, 100)
y = n/np.sqrt(2*np.pi)*np.exp(-0.5*x*x)
plt.plot(x, y, 'r')
plt.xlabel("$x_i$")
plt.ylabel("number in each bin")
plt.subplots_adjust(wspace=0.4)
plt.show()
