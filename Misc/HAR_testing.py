import numpy as np

def fill_to_x(x, g, b):
    istar = g // ((1-g) + 0.0001)
    eps = 0.0001
    val = x
    weight = 0
    i=1
    while val > eps:
        if (1/(i+1)) < val:
            val -= (1/(i+1))
            weight += (1/i)
            if i > istar:
                weight += ((1/(i+1)) - (g/i)) * b 
        i += 1
        
    return weight


def fill_to_x_thresholded(x, g, b, tau):

    if tau >= 1 - g:
        tau = 1 - g

    istar = g / tau
    eps = 0.0001
    val = x
    weight = 0
    i=1
    while val > eps:
        if ((g+tau)/(i)) < val:
            # print(val)
            # print((g+tau)/(i+1), i, g, tau,weight)
            if tau < 1/(i*b) and i <= istar:
                val -= ((g+tau)/(i + 1))
                weight += (1/i)

            else: 
                val -= ((g+tau)/(i))
                weight +=  (tau*b + 1) / i
             
        i += 1
        
    return weight


g = 1/3
b = 10/3
tau = 1/b

print(fill_to_x_thresholded(g, g, b, tau))

number = 100
bmax = 20
gmax = 1
gs = np.round(np.arange(gmax, 0, -gmax/number), 4)
bs = np.round(np.arange(0, bmax, bmax/number), 4)


cost_per_bin = np.zeros((number,number))
CRs = np.zeros((number,number))

for i,g in enumerate(gs):
    for j,b in enumerate(bs):
        if g * b > 1:
            max_weight = fill_to_x_thresholded(g, g, b, 1/b)
            cost_per_bin[i,j] = 1
            CRs[i,j] = max_weight
        else:
            max_weight = fill_to_x(1, g, b)
            cost_per_bin[i,j] = 1 + (1-g)*b
            CRs[i,j] = max_weight / cost_per_bin[i,j]

print(CRs.min())

import matplotlib.pyplot as plt
import seaborn as sns

ax = plt.axes()
sns.set()

sns.heatmap(CRs, linewidth=0, xticklabels=[], yticklabels=[],annot=False, ax=ax)
plt.xlabel("B (max = " + str(bmax)+ ")")
plt.ylabel("G")
plt.title("CR of thresholded GHAR algorithm (non thresholded for Gb < 1)")
plt.show()