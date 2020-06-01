"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Finding Random numbers using numpy.random.rand

"""

import numpy as np
import matplotlib.pyplot as plt
import time

n = 10000
strt_time = time.time()
y = np.random.rand(n)
end_time = time.time()
print(end_time-strt_time)
plt.hist(y)
plt.title("Random numbers using numpy.random.rand")
plt.xlabel("$x_i$")
plt.ylabel("number in each bin")
x = np.linspace(0, 1, 100)
y = n/10*np.ones(len(x))
plt.plot(x, y, 'r')
plt.show()
