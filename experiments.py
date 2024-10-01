import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import math
from matplotlib.colors import LinearSegmentedColormap
import cooper_ghar




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

def gaaf_s(arr, g, b):
    if g + 1/b >= 1:
        return 1.7 * opt_s(arr, g, b)
    elif g*b >= 1:
        return 2*np.sum(arr) / (g+ (1/b))
    else:
        return opt_s(arr, g, b)
    

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

def green_worst_fit(arr,g,b):
    cost = 1
    bins = [0]
    threshold = min(g + (1/b), 1)
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

def best_fit(arr,g,b, threshold=1):
    cost = 1
    bins = [0]
    if g*b <= 1:
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

def first_fit(arr,g,b, threshold=1):
    cost = 1
    bins = [0]
    if g*b <= 1:
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

def next_fit(arr, g, b, threshold=1):
    curr = 1
    cost = 0
    if g*b <= 1:
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


def gamma(i, add_one_denom=False):

    if i == -1:
        if add_one_denom:
            return 1
        return 1.691
    if i == 0:
        if add_one_denom:
            return 0.5
        return 1
    
    if add_one_denom:
        return gamma(i) - gamma(i+1)
    
    else:
        return  gamma(i-1)**2 / (gamma(i-1) + 1)
    

def sum_gamma(stop, add_one_denom=False):
    sum = 0

    for i in range(stop):
        sum += gamma(i, add_one_denom)

    return sum

def Har_CR(g, b):

    if g == 1:
        return 1.691
    else:
        i_star = min(5,math.floor(g/(1-g)))

    numerator = b * (1 - sum_gamma(i_star, add_one_denom=True) - g*(1.691 - sum_gamma(i_star))) + 1.691
    if abs(g - 0.5) < 0.1:
        print("black part of numerator:",(numerator - 1.691)/b)
        print("value of g:", g)
    denom = b*(1-g) + 1

    return numerator/denom 

def combined_cr_UB(g,b):
    if g*b < 1:
        return Har_CR(g,b)
    if g*b > 17/3:
        return 1.7
    if g + 1/b > 1:
        return 1.7 * g * (3/17 * (1-g)*b + 1)
    else:
        return 2*g/(g + 1/b)


def Har(arr, g, b, max_i=100):
    if g == 1:
        i_star = max_i
    else:
        i_star = min(math.floor(g/(1-g)), max_i)
    cost = 0

    print(i_star)
    print(max_i)
    print(i_star + max_i)

    bins = np.zeros((max_i + i_star, 3)) # Each bin is a tuple (weight, count, i)
    for i in range(1, max_i+1):
        if i <= i_star:
            bins[2*i - 1][2] = i
            bins[2*i - 2][2] = i
        else:
            bins[i + i_star - 1][2] = i

    for item in arr:
        item_index = -1
        for i in range(1, max_i+1):
            if i <= i_star:
                if (1/(i+1)) <= item and item < (g/i):
                    item_index = 2*i - 1
                elif (g/i) <= item and item < (1/i):
                    item_index = 2*i - 2
            elif (1/(i+1)) <= item and item < (1/i):
                item_index = i + i_star - 1
        
        new_bin, item_cost = add_item_har(bins[item_index], item, g, b)
        bins[item_index] = new_bin
        cost += item_cost

    return cost, bins


def add_item_har(bin, item, g, b):
    cost = 0
    if bin[0] == 0:
        bin[0] = item
        bin[1] = 1
        cost += max(item - g, 0) * b + 1
        return bin, cost
    
    if bin[1] == bin[2]:
        cost += 1
        bin[1] = 1
        bin[0] = item

    else:
        cost += max(item - max(g - bin[0], 0), 0) * b
        bin[0] += item
        bin[1] += 1

    return bin, cost

def worst_fit(arr,g,b, threshold=1):
    cost = 1
    bins = [0]

    if g*b <= 1:
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

gs = np.round(np.arange(gmax, 0, -gmax/number), 4) + 0.01
bs = np.round(np.arange(0, bmax, bmax/number), 4) + 0.01


num_threshs = 100
threshs = np.round(np.arange(0, 1, 1/num_threshs), 4)

opt_s_store = np.zeros((number,number))
nf_s_store = np.zeros((number,number))
gaaf_s_store = np.zeros((number,number))
green_next_fit_store = np.zeros((number,number))
green_worst_fit_store = np.zeros((number,number))
green_best_fit_store = np.zeros((number,number))
green_first_fit_store = np.zeros((number,number))

next_fit_store = np.zeros((number,number))
worst_fit_store = np.zeros((number,number))
best_fit_store = np.zeros((number,number))
first_fit_store = np.zeros((number,number))

HAR_store = np.zeros((number, number))
GHAR_store = np.zeros((number, number))

alg_store = np.zeros((10, number**2))
gb = np.zeros((number**2))

whose_best = np.zeros((number,number))
best_alg_store = np.zeros((number,number))
best_trad_alg_store = np.zeros((number,number))
best_green_alg_store = np.zeros((number,number))

best_bf_thresh = np.zeros((number,number))
best_ff_thresh = np.zeros((number,number))
best_wf_thresh = np.zeros((number,number))

diff_from_g = np.zeros((3,number,number))

test_g = np.zeros((number,number))
test_b = np.zeros((number,number))

combined_cr_UB_store = np.zeros((number,number))

best_fit_cases = [[],[]]
green_best_fit_cases = [[], []]



for i,g in enumerate(tqdm(gs)):
    for j,b in enumerate(bs):
        example = np.random.rand(100)
        if g*b >= 1:
            example *= g
        
        # combined_cr_UB_store[i,j] = combined_cr_UB(g,b)
        # HAR_store[i,j], _ = Har(example, g, b, max_i=100)
        GHAR_store[i,j] = cooper_ghar.ghar_CR_calc(g,b,1/b,8, 15)[0]

        # opt_s_store[i,j] = opt_s(example, g, b)
        # gaaf_s_store[i,j] = gaaf_s(example, g, b)
        # # nf_s_store[i,j] = nf_s(example, g, b)
        # green_next_fit_store[i,j] = green_next_fit(example, g, b)
        # green_worst_fit_store[i,j] = green_worst_fit(example, g, b)
        # green_best_fit_store[i,j] = green_best_fit(example, g, b)
        # green_first_fit_store[i,j] = green_first_fit(example, g, b)
        # next_fit_store[i,j] = next_fit(example, g, b, threshold=g)
        # worst_fit_store[i,j] = worst_fit(example, g, b, threshold=g)
        # best_fit_store[i,j] = best_fit(example, g, b, threshold=g)
        # first_fit_store[i,j] = first_fit(example, g, b, threshold=g)


        # for k,thresh in enumerate(threshs):
        #     wf_k = worst_fit(example, g, b, threshold=thresh)
        #     if worst_fit_store[i,j] == 0 or wf_k < worst_fit_store[i,j]:
        #         worst_fit_store[i,j] = wf_k
        #         best_wf_thresh[i,j] = k/num_threshs
        #         diff_from_g[0,i,j] = g - best_wf_thresh[i,j]
            
        #     bf_k = best_fit(example, g, b, threshold=thresh)
        #     if best_fit_store[i,j] == 0 or bf_k < best_fit_store[i,j]:
        #         best_fit_store[i,j] = bf_k
        #         best_bf_thresh[i,j] = k/num_threshs
        #         diff_from_g[1,i,j] = g - best_bf_thresh[i,j]

        #     ff_k = first_fit(example, g, b, threshold=thresh)
        #     if first_fit_store[i,j] == 0 or ff_k < first_fit_store[i,j]:
        #         first_fit_store[i,j] = ff_k
        #         best_ff_thresh[i,j] = k/num_threshs
        #         diff_from_g[2,i,j] = g - best_ff_thresh[i,j]



        # green_costs = np.array([green_next_fit_store[i,j], green_worst_fit_store[i,j], green_best_fit_store[i,j], green_first_fit_store[i,j]])
        # trad_costs = np.array([next_fit_store[i,j], worst_fit_store[i,j], best_fit_store[i,j], first_fit_store[i,j]])
        # all_costs = np.array([green_next_fit_store[i,j], green_worst_fit_store[i,j], green_best_fit_store[i,j], green_first_fit_store[i,j], next_fit_store[i,j], worst_fit_store[i,j], best_fit_store[i,j], first_fit_store[i,j] ])

        # whose_best[i,j] = np.argmin(all_costs)
        # best_alg_store[i,j] = np.min(all_costs)
        # best_green_alg_store[i,j] = np.min(green_costs)
        # best_trad_alg_store[i,j] = np.min(trad_costs)

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

    ## B=1/g graphs ##

    # plt.title("Costs of each alg when g = 1/b - OPT")
    # plt.xlabel("Value of G")
    # plt.ylabel("Cost")
    # # plt.plot(gs.flatten(), opt_s_store[:,0], label="OPT(S)")
    # plt.plot(gs.flatten(), best_fit_store[:,0] - opt_s_store[:,0], label="BF")
    # plt.plot(gs.flatten(), green_best_fit_store[:,0] - opt_s_store[:,0], label="GBF")
    # plt.plot(gs.flatten(), worst_fit_store[:,0] - opt_s_store[:,0], label="WF")
    # plt.plot(gs.flatten(), green_worst_fit_store[:,0] - opt_s_store[:,0], label="GWF")
    # plt.plot(gs.flatten(), first_fit_store[:,0] - opt_s_store[:,0], label="FF")
    # plt.plot(gs.flatten(), green_first_fit_store[:,0] - opt_s_store[:,0], label="GFF")
    # plt.legend()
    # plt.show()

ax = sns.heatmap(GHAR_store, linewidth=0, xticklabels=[], yticklabels=[])
plt.title("CR of GHAR (mimi)")
plt.xlabel("Value of B")
plt.ylabel("Value of G")
plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
plt.legend()
plt.show()

# ax = sns.heatmap(HAR_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("CR of HAR")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
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

# ax = sns.heatmap(best_bf_thresh, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best Threshold for BF")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(best_wf_thresh, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best Threshold for WF")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(best_ff_thresh, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best Threshold for FF")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()

# ax = sns.heatmap(best_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best cost for BF over all thresholds")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(worst_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best cost for WF over all thresholds")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(first_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best cost for FF over all thresholds")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()

# ax = sns.heatmap(diff_from_g[0], linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best threshold for BF - G")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(diff_from_g[1], linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best threshold for WF - G")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(diff_from_g[2], linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best threshold for FF - G")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()

# ax = sns.heatmap((green_best_fit_store < best_fit_store), linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Cost of instances where GBF < BF for varied G and B")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.show()

# myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
# cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
# cmap = sns.color_palette("deep", 8)

# ax = sns.heatmap(whose_best, cmap=cmap, linewidth=0, xticklabels=[], yticklabels=[])
# colorbar = ax.collections[0].colorbar
# colorbar.set_ticks(np.array([0, 1, 2, 3, 4, 5, 6, 7]))
# colorbar.set_ticklabels(['GNF', 'GWF-m', 'GBF', 'GFF', 'NF', 'WF', 'BF', 'FF'])
# r = colorbar.vmax - colorbar.vmin
# colorbar.set_ticks([colorbar.vmin + 0.5 * r / (8) + r * i / (8) for i in range(8)])
# plt.title("Alg with minimum value (modified trad with threshold 1-g)")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()



    # ## DIFFERENCE GRAPHS ##

# ax = sns.heatmap(combined_cr_UB_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("CR Upper bound for combined algorithm (HAR + OPT AAF)")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()

# ax = sns.heatmap(gaaf_s_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("GAAF(S) ")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(gaaf_s_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("GAAF CR upper bound ")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(gaaf_s_store/opt_s_store - 1.7, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("GAF CR upper bound - 1.7 ")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(((gaaf_s_store/opt_s_store - 1.7)<0) + ((gaaf_s_store/opt_s_store - 1.54)<0), linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Region where GAF CR is beter than AF CR ")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(best_green_alg_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best green alg CR ")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(best_trad_alg_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best trad alg CR ")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
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
