"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Testing randomness of distribution

"""

import numpy as np
import scipy.stats as st


def randomness(chi):          # returns randomness baded on chi^2 value
    p = (1.0 - st.chi2.cdf(chi, 10.0)) * 100
    print(p)
    if 0 < p < 1 or 99 < p < 100:
        print("not suffiently random")
    elif 1 < p < 5 or 95 < p < 99:
        print("suspect")
    elif 5 < p < 10 or 90 < p < 95:
        print("almost suspect")
    else:
        print("suffiently random")


obs1 = np.array([4, 10, 10, 13, 20, 18, 18, 11, 13, 14, 13])
obs2 = np.array([3, 7, 11, 15, 19, 24, 21, 17, 13, 9, 5])
prob = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])/36         # Propability of diferrent values
exp_dis = np.sum(obs1) * prob                                 # Expected values prop * num_of_throws
V1 = ((obs1-exp_dis)**2)/exp_dis
chi1 = np.sum(V1)
V2 = ((obs2-exp_dis)**2)/exp_dis
chi2 = np.sum(V2)
print("Randomness of 1")
randomness(chi1)
print("Ramdomness of 2")
randomness(chi2)
