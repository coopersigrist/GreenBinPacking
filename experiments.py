import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm




def opt_s(arr, g, b):
    s = np.sum(arr)
    if g*b >= 1:
        return s/g
    else:
        return s + s*b*(1-g)
    
def nf_s(arr, g, b):
    s = np.sum(arr)
    threshold = min(g + 1/b, 1)
    return (3*s)/threshold

def green_next_fit(arr, g, b):
    curr = 0
    cost = 0
    threshold = min(g + 1/b, 1)
    for item in arr:
        if curr + item > threshold:
            cost += max((curr - g), 0)*b + 1
            curr = item
        else:
            curr += item
    
    return cost

def green_worst_fit(arr,g,b):
    cost = 0
    bins = [0]
    threshold = min(g + 1/b, 1)
    for item in arr:
        mini = min(bins)
        if mini + item <= threshold:
            bins[bins.index(mini)] += item
        else:
            cost += 1
            bins.append(item)
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost

def green_best_fit(arr,g,b):
    cost = 0
    bins = [0]
    threshold = min(g + 1/b, 1)
    for item in arr:
        allowed = list(filter(lambda n: n < (threshold - item), bins))
        if len(allowed) == 0:
            cost += 1
            bins.append(item)
        else:
            bins[bins.index(max(allowed))] += item
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost

def green_first_fit(arr,g,b):
    cost = 0
    bins = [0]
    threshold = min(g + 1/b, 1)
    for item in arr:
        allowed = list(filter(lambda n: n < (threshold - item), bins))
        if len(allowed) == 0:
            cost += 1
            bins.append(item)
        else:
            bins[bins.index(allowed[0])] += item
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost


number = 100
gs = np.round(np.arange(0, 1, 1/number), 4)
bs = np.round(np.arange(0, 2, 2/number), 4)

opt_s_store = np.zeros((number,number))
nf_s_store = np.zeros((number,number))
green_next_fit_store = np.zeros((number,number))
green_worst_fit_store = np.zeros((number,number))
green_best_fit_store = np.zeros((number,number))
green_first_fit_store = np.zeros((number,number))

example = np.random.rand(100)

for i,g in enumerate(gs):
    for j,b in enumerate(bs):
        opt_s_store[i,j] = opt_s(example, g, b)
        nf_s_store[i,j] = nf_s(example, g, b)
        green_next_fit_store[i,j] = green_next_fit(example, g, b)
        green_worst_fit_store[i,j] = green_worst_fit(example, g, b)
        green_best_fit_store[i,j] = green_best_fit(example, g, b)
        green_first_fit_store[i,j] = green_first_fit(example, g, b)

# vmin = min(np.min(opt_s_store), np.min(nf_s_store), np.min(green_next_fit_store))
# vmax = max(np.max(opt_s_store), np.max(nf_s_store), np.max(green_next_fit_store))

# plt.subplot(1, 3, 1)

# ax = sns.heatmap(opt_s_store, linewidth=0, xticklabels=[], yticklabels=[], vmin=vmin, vmax=vmax, cbar=False)
# plt.title("values of OPT(S) for varied G and B")
# plt.xlabel("Value of G")
# plt.ylabel("Value of B")



# plt.subplot(1, 3, 2)

# ax = sns.heatmap(nf_s_store, linewidth=0, xticklabels=[], yticklabels=[], vmin=vmin, vmax=vmax, cbar=False)
# plt.title("values of NF(S) for varied G and B")
# plt.xlabel("Value of G")
# plt.ylabel("Value of B")



# plt.subplot(1, 3, 3)

# ax = sns.heatmap(green_next_fit_store, linewidth=0, xticklabels=[], yticklabels=[], vmin=vmin, vmax=vmax)
# plt.title("Cost of Green next fit for varied G and B")
# plt.xlabel("Value of G")
# plt.ylabel("Value of B")
# plt.show()

# plt.subplot(1, 2, 1)

# ax = sns.heatmap(opt_s_store, linewidth=0, xticklabels=[], yticklabels=[], vmin=vmin, vmax=vmax, cbar=False)
# plt.title("values of OPT(S) for varied G and B")
# plt.xlabel("Value of G")
# plt.ylabel("Value of B")

# plt.subplot(1, 2, 2)

# ax = sns.heatmap(green_next_fit_store, linewidth=0, xticklabels=[], yticklabels=[], vmin=vmin, vmax=vmax)
# plt.title("Cost of Green worst fit for varied G and B")
# plt.xlabel("Value of G")
# plt.ylabel("Value of B")
# plt.show()

# ax = sns.heatmap(nf_s_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("CR NF(S) / OPT(S) for varied G and B")
# plt.xlabel("Value of G")
# plt.ylabel("Value of B")
# plt.plot((number/(2 * gs)), number - number*gs, label="G=1/B" )
# plt.legend()
# plt.show()

# ax = sns.heatmap(nf_s_store/green_next_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Cost of NF(S) / Real Green Next Fit ")
# plt.xlabel("Value of G")
# plt.ylabel("Value of B")
# plt.plot((number/(2 * gs)), number - number*gs, label="G=1/B" )
# plt.legend()
# plt.show()

ax = sns.heatmap(green_best_fit_store/green_worst_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
plt.title("Cost of GBF / GWF ")
plt.xlabel("Value of G")
plt.ylabel("Value of B")
plt.show()

ax = sns.heatmap(green_first_fit_store/green_worst_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
plt.title("Cost of GFF / GWF ")
plt.xlabel("Value of G")
plt.ylabel("Value of B")
plt.show()

# ax = plt.subplot(1, 2, 1)

# sns.heatmap(green_next_fit_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[], ax=ax)
# plt.title("CR Green next fit / OPT(S) for varied G and B")
# plt.xlabel("Value of G")
# plt.ylabel("Value of B")
# plt.plot((number/(2 * gs)), number - number*gs, label="G=1/B" )
# plt.legend()



# plt.subplot(1, 2, 2)

# ax = sns.heatmap(green_worst_fit_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("CR Green worst fit / OPT(S) for varied G and B")
# plt.xlabel("Value of G")
# plt.ylabel("Value of B")
# plt.plot((number/(2 * gs)), number - number*gs, label="G=1/B" )
# plt.show()

# plt.imshow(opt_s_store, cmap='hot', interpolation='nearest')

# plt.show()
