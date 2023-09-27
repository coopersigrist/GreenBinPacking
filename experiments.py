import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap




def opt_s(arr, g, b):
    s = np.sum(arr)
    if g*b >= 1:
        return s/g
    else:
        return s + s*b*(1-g)
    
def nf_s(arr, g, b):
    s = np.sum(arr)
    threshold = min(g + (1/b), 1)
    return (3*s)/threshold

def green_next_fit(arr, g, b):
    curr = 0
    cost = 1
    threshold = min(g + (1/b), 1)
    for item in arr:
        if curr + item > threshold:
            cost += max((curr - g), 0)*b + 1
            curr = item
        else:
            curr += item
    
    return cost

def green_worst_fit(arr,g,b,gamma):
    cost = 1
    bins = [0]

    if gamma == 0:
        factor = 0
    else:
        dist = np.sqrt(((b - (1/(1-g)))**2 + (g - (1/b))**2))
        factor = dist/(dist + gamma)

    threshold = min(g + (1/b) - (factor/b), 1)
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
    cost = 1
    bins = [0]
    threshold = min(g + (1/b), 1)
    for item in arr:
        allowed = list(filter(lambda n: n <= (threshold - item), bins))
        if len(allowed) == 0:
            cost += 1
            bins.append(item)
        else:
            bins[bins.index(max(allowed))] += item
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost

def green_first_fit(arr,g,b):
    cost = 1
    bins = [0]
    threshold = min(g + (1/b), 1)
    for item in arr:
        allowed = list(filter(lambda n: n <= (threshold - item), bins))
        if len(allowed) == 0:
            cost += 1
            bins.append(item)
        else:
            bins[bins.index(allowed[0])] += item
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost

def best_fit(arr,g,b):
    cost = 1
    bins = [0]
    if g*b < 1:
        threshold = 1
    else:
        threshold = g
    for item in arr:
        allowed = list(filter(lambda n: n <= (threshold - item), bins))
        if len(allowed) == 0:
            cost += 1
            bins.append(item)
        else:
            bins[bins.index(max(allowed))] += item
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost
    
    return cost

def first_fit(arr,g,b):
    cost = 1
    bins = [0]
    if g*b < 1:
        threshold = 1
    else:
        threshold = g
    for item in arr:
        allowed = list(filter(lambda n: n <= (threshold - item), bins))
        if len(allowed) == 0:
            cost += 1
            bins.append(item)
        else:
            bins[bins.index(allowed[0])] += item
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost

def next_fit(arr, g, b):
    curr = 1
    cost = 0
    if g*b < 1:
        threshold = 1
    else:
        threshold = g
    for item in arr:
        if curr + item > threshold:
            cost += max((curr - g), 0)*b + 1
            curr = item
        else:
            curr += item
    
    return cost

def worst_fit(arr,g,b):
    cost = 1
    bins = [0]
    if g*b < 1:
        threshold = 1
    else:
        threshold = g
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

number = 100
bmax = 10
gmax = 1
gs = np.round(np.arange(0, gmax, gmax/number), 4)
bs = np.round(np.arange(0, bmax, bmax/number), 4)

opt_s_store = np.zeros((number,number))
nf_s_store = np.zeros((number,number))
green_next_fit_store = np.zeros((number,number))
green_worst_fit_store = np.zeros((number,number))
green_best_fit_store = np.zeros((number,number))
green_first_fit_store = np.zeros((number,number))

next_fit_store = np.zeros((number,number))
worst_fit_store = np.zeros((number,number))
best_fit_store = np.zeros((number,number))
first_fit_store = np.zeros((number,number))

alg_store = np.zeros((10, number**2))
gb = np.zeros((number**2))

whose_best = np.zeros((number,number))
best_alg_store = np.zeros((number,number))
best_trad_alg_store = np.zeros((number,number))
best_green_alg_store = np.zeros((number,number))

best_fit_cases = [[],[]]
green_best_fit_cases = [[], []]

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for i,g in enumerate(tqdm(gs)):
        for j,b in enumerate(bs):
            example = np.random.rand(100)
            example *= min(1, g + (1/b))
            opt_s_store[i,j] = opt_s(example, g, b)
            nf_s_store[i,j] = nf_s(example, g, b)
            green_next_fit_store[i,j] = green_next_fit(example, g, b)
            green_worst_fit_store[i,j] = green_worst_fit(example, g, b, gamma=gamma)
            green_best_fit_store[i,j] = green_best_fit(example, g, b)
            green_first_fit_store[i,j] = green_first_fit(example, g, b)

            next_fit_store[i,j] = next_fit(example, g, b)
            worst_fit_store[i,j] = worst_fit(example, g, b)
            best_fit_store[i,j] = best_fit(example, g, b)
            first_fit_store[i,j] = first_fit(example, g, b)

            green_costs = np.array([green_next_fit_store[i,j], green_worst_fit_store[i,j], green_best_fit_store[i,j], green_first_fit_store[i,j]])
            trad_costs = np.array([next_fit_store[i,j], worst_fit_store[i,j], best_fit_store[i,j], first_fit_store[i,j]])
            all_costs = np.array([green_next_fit_store[i,j], green_worst_fit_store[i,j], green_best_fit_store[i,j], green_first_fit_store[i,j], next_fit_store[i,j], worst_fit_store[i,j], best_fit_store[i,j], first_fit_store[i,j] ])

            whose_best[i,j] = np.argmin(all_costs)
            best_alg_store[i,j] = np.min(all_costs)
            best_green_alg_store[i,j] = np.min(green_costs)
            best_trad_alg_store[i,j] = np.min(trad_costs)

            # if best_fit_store[i,j] < green_best_fit_store[i,j]:
            #     best_fit_cases[0].append(b * g)
            #     best_fit_cases[1].append(best_fit_store[i,j])
            # else:
            #     green_best_fit_cases[0].append(b*g)
            #     green_best_fit_cases[1].append(green_best_fit_store[i,j])

            # alg_store[0, i*number + j] = green_next_fit_store[i,j]
            # alg_store[1, i*number + j] = green_worst_fit_store[i,j]
            # alg_store[2, i*number + j] = green_best_fit_store[i,j]
            # alg_store[3, i*number + j] = green_first_fit_store[i,j]
            # alg_store[4, i*number + j] = opt_s_store[i,j]
            # alg_store[5, i*number + j] = nf_s_store[i,j]
            # alg_store[6, i*number + j] = next_fit_store[i,j]
            # alg_store[7, i*number + j] = worst_fit_store[i,j]
            # alg_store[8, i*number + j] = best_fit_store[i,j]
            # alg_store[9, i*number + j] = first_fit_store[i,j]


            # gb[i*number + j] = g * b


    ## COST GRAPHS ##
    # plt.scatter(gb, alg_store[2], label="GBF")
    # plt.scatter(gb, alg_store[3], label="GFF")
    # plt.scatter(gb, alg_store[8], label="BF")
    # plt.scatter(gb, alg_store[9], label="FF")

    # plt.scatter(gb, alg_store[4], label="OPT(S)")

    # plt.title("cost based on G * Beta")
    # plt.xlabel("G*Beta")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.show()

    # ax = sns.heatmap((best_alg_store), linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("Minimum costs between all basic (green and standard) algorithms")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
    # plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
    # plt.legend()
    # plt.show()
    # plt.show()

    ## RATIO GRAPHS ##

    # ax = sns.heatmap((best_alg_store / opt_s_store), linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("ratio of minimum cost alg to OPT")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
    # plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
    # plt.legend()
    # plt.show()
    # plt.show()

    # # plt.scatter(gb, alg_store[5], label="NF(S)")
    # # plt.scatter(gb, alg_store[0]/alg_store[4], label="GNF / OPT(S)")
    # # plt.scatter(gb, alg_store[1]/alg_store[4], label="GWF / OPT(S)")
    # plt.scatter(gb, alg_store[2]/alg_store[4], label="GBF / OPT(S)")
    # plt.scatter(gb, alg_store[3]/alg_store[4], label="GFF / OPT(S)")
    # # plt.scatter(gb, alg_store[6]/alg_store[4], label="NF / OPT(S)")
    # # plt.scatter(gb, alg_store[7]/alg_store[4], label="WF / OPT(S)")
    # plt.scatter(gb, alg_store[8]/alg_store[4], label="BF / OPT(S)")
    # plt.scatter(gb, alg_store[9]/alg_store[4], label="FF / OPT(S)")

    # # plt.scatter(gb, alg_store[4], label="OPT(S)")

    # plt.title("cost based on G * Beta")
    # plt.xlabel("G*Beta")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.show()

    ## BEST ALG GRAPHS ##

    # plt.scatter(best_fit_cases[0],best_fit_cases[1] , label="BF < GBF")
    # plt.scatter(green_best_fit_cases[0], green_best_fit_cases[1] , label="BF > GBF")
    # plt.title("cost based on G * Beta")
    # plt.xlabel("G*Beta")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.show()

    # ax = sns.heatmap(green_best_fit_store/best_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("Cost of GBF/BF for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # # plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
    # # plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
    # # plt.legend()
    # plt.show()

    # ax = sns.heatmap((green_best_fit_store < best_fit_store), linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("Cost of instances where GBF < BF for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.show()

    # myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
    # cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    cmap = sns.color_palette("deep", 8)

    ax = sns.heatmap(whose_best, cmap=cmap, linewidth=0, xticklabels=[], yticklabels=[])
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    colorbar.set_ticklabels(['GNF', 'GWF-m', 'GBF', 'GFF', 'NF', 'WF', 'BF', 'FF'])
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / (8) + r * i / (8) for i in range(8)])
    plt.title("Alg with minimum value (modified GWF with GAMMA = " +str(gamma)+")")
    plt.xlabel("Value of B")
    plt.ylabel("Value of G")
    plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
    plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
    plt.legend()
    plt.show()

    # ## DIFFERENCE GRAPHS ##

    # ax = sns.heatmap(best_trad_alg_store - best_green_alg_store, linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("Best Trad (threshold=g) cost - best green cost ")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
    # plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
    # plt.legend()
    # plt.show()

    # plt.scatter(gb, alg_store[8] - alg_store[2], label="BF - GBF")
    # plt.scatter(gb, alg_store[9] - alg_store[3], label="FF - GFF" )

    # plt.title("cost based on G * Beta")
    # plt.xlabel("G*Beta")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.show()

    # vmin = min(np.min(opt_s_store), np.min(nf_s_store), np.min(green_next_fit_store))
    # vmax = max(np.max(opt_s_store), np.max(nf_s_store), np.max(green_next_fit_store))

    # plt.subplot(1, 3, 1)

    # ax = sns.heatmap(green_next_fit_store/nf_s_store, linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("GNF/NF(S) for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.show()


    # plt.subplot(1, 3, 2)

    # ax = sns.heatmap(nf_s_store, linewidth=0, xticklabels=[], yticklabels=[], vmin=vmin, vmax=vmax, cbar=False)
    # plt.title("values of NF(S) for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")



    # plt.subplot(1, 3, 3)

    # ax = sns.heatmap(green_next_fit_store, linewidth=0, xticklabels=[], yticklabels=[], vmin=vmin, vmax=vmax)
    # plt.title("Cost of Green next fit for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.show()

    # plt.subplot(1, 2, 1)

    # ax = sns.heatmap(opt_s_store, linewidth=0, xticklabels=[], yticklabels=[], vmin=vmin, vmax=vmax, cbar=False)
    # plt.title("values of OPT(S) for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")

    # plt.subplot(1, 2, 2)

    # ax = sns.heatmap(green_next_fit_store, linewidth=0, xticklabels=[], yticklabels=[], vmin=vmin, vmax=vmax)
    # plt.title("Cost of Green worst fit for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.show()

    # ax = sns.heatmap(nf_s_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("CR NF(S) / OPT(S) for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.plot((number/(2 * gs)), number - number*gs, label="G=1/B" )
    # plt.legend()
    # plt.show()

    # ax = sns.heatmap(nf_s_store/green_next_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("Cost of NF(S) / Real Green Next Fit ")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.plot((number/(2 * gs)), number - number*gs, label="G=1/B" )
    # plt.legend()
    # plt.show()

    # ax = sns.heatmap(green_best_fit_store/green_worst_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("Cost of GBF / GWF ")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.show()

    # ax = sns.heatmap(green_first_fit_store/green_worst_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("Cost of GFF / GWF ")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.show()

    # ax = sns.heatmap(green_first_fit_store/green_best_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("Cost of GFF / GBF ")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.show()

    # ax = plt.subplot(1, 2, 1)

    # sns.heatmap(green_next_fit_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[], ax=ax)
    # plt.title("CR Green next fit / OPT(S) for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.plot((number/(2 * gs)), number - number*gs, label="G=1/B" )
    # plt.legend()



    # plt.subplot(1, 2, 2)

    # ax = sns.heatmap(green_worst_fit_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("CR Green worst fit / OPT(S) for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.plot((number/(2 * gs)), number - number*gs, label="G=1/B" )
    # plt.show()

    # plt.imshow(opt_s_store, cmap='hot', interpolation='nearest')

    # plt.show()
