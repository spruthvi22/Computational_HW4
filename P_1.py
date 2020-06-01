"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Finding Random numbers using linear congruential generator

"""

import numpy as np
import matplotlib.pyplot as plt
import time


def LCG(n, seed=1): 
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


n = 10000
strt_time = time.time()
y = LCG(n)
end_time = time.time()
print(end_time-strt_time)
plt.hist(y)
plt.title("Random numbers using Linear Congruential Generator")
plt.xlabel("$x_i$")
plt.ylabel("number in each bin")
x = np.linspace(0, 1, 100)
y = n/10*np.ones(len(x))
plt.plot(x, y, 'r')
plt.show()
