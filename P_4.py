"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Plotting density histogram of c-code

"""

import numpy as np
import matplotlib.pyplot as plt


# Plotting the uniform and transformed- exponential distributions
n = 10000
p4 = np.genfromtxt('P_4.csv', delimiter=',')
plt.subplot(1, 2, 1, title="Uniform distribution")
plt.hist(p4[:, 0])
x = np.linspace(0, 1, 100)
y = n/10*np.ones(len(x))
plt.plot(x, y, 'r')
plt.xlabel("$x_i$")
plt.ylabel("number in each bin")
plt.subplot(1, 2, 2, title="Exponential Distribution")
plt.hist(p4[:, 1], range=[0, 10])
x = np.linspace(0, 10, 100)
y = n*0.5*np.exp(-0.5*x)
plt.plot(x, y, 'r')
plt.xlabel("$x_i$")
plt.ylabel("number in each bin")
plt.subplots_adjust(wspace=0.4)
plt.show()
