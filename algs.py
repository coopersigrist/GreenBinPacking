import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import math
from matplotlib.colors import LinearSegmentedColormap

# Returns the OPT lower bound for a sequence based on its volume
def opt_s(arr, g, b):
    s = np.sum(arr)
    if g*b >= 1:
        return s/g
    else:
        return s + s*b*(1-g)
    
def opt_smart(arr, g, b):
    small = list(filter(lambda x: x <= g, arr))
    large = list(filter(lambda x: x > g, arr))
    s = np.sum(small)/g
    l = list(map(lambda x: 1+b*(x-g), large))
    return s + np.sum(l)

    
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
    
# def green_next_fit(arr, g, b):
#     curr = 0
#     cost = 1
#     threshold = min(g + (1/b), 1)
#     for item in arr:
#         if curr + item > threshold:
#             cost += max((curr - g), 0)*b + 1
#             curr = item
#         else:
#             curr += item
    
#     return cost

# def green_worst_fit(arr,g,b):
#     cost = 1
#     bins = [0]
#     threshold = min(g + (1/b), 1)
#     for item in arr:
#         mini = min(bins)
#         if mini + item <= threshold:
#             bins[bins.index(mini)] += item
#         else:
#             cost += 1
#             bins.append(item)
            
#     for bin in bins:
#         cost += max(bin - g, 0) * b
    
#     return cost

# def green_best_fit(arr,g,b):
#     cost = 1
#     bins = [0]
#     threshold = min(g + (1/b), 1)
#     for item in arr:
#         allowed = list(filter(lambda n: n <= (threshold - item), bins))
#         if len(allowed) == 0:
#             cost += 1
#             bins.append(item)
#         else:
#             bins[bins.index(max(allowed))] += item
            
#     for bin in bins:
#         cost += max(bin - g, 0) * b
    
#     return cost

# def green_first_fit(arr,g,b):
#     cost = 1
#     bins = [0]
#     threshold = min(g + (1/b), 1)
#     for item in arr:
#         allowed = list(filter(lambda n: n <= (threshold - item), bins))
#         if len(allowed) == 0:
#             cost += 1
#             bins.append(item)
#         else:
#             bins[bins.index(allowed[0])] += item
            
#     for bin in bins:
#         cost += max(bin - g, 0) * b
    
#     return cost

def best_fit(arr,g,b, threshold=1):
    cost = 1
    bins = [0]

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
    

def first_fit(arr,g,b, threshold=1):
    cost = 1
    bins = [0]
    
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
    curr = 0
    cost = 0

    for item in arr:
        if curr + item > threshold:
            cost += max((curr - g), 0)*b + 1
            curr = item
        else:
            curr += item

    #Add cost of final bin
    cost += max((curr - g), 0)*b + 1

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

def Har(arr, g, b, max_i=100):
    if g == 1:
        i_star = max_i
    else:
        i_star = min(math.floor(g/(1-g)), max_i)
    cost = 0

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

def Ghar(arr, g, b, tau, imax=10):

    bins = np.zeros((imax+1, 3)) # each bin is a truple of total cost of i-type bins, count of items in current opened i-type bin, and amount filled
    for item in arr:
        i = 0
        while item < (g + tau) / (i + 1) and i < imax : 
            i += 1
        bins[i][1] += 1 # increase count of number of items in bin
        bins[i][2] += item # add item to bin
        if  bins[i][1] >= i:
            bins[i][0] += 1 + max(bins[i][2] - g, 0)*b # adds the cost of the filled bin to total
            bins[i][1] = 0 # resets count of bin
            bins[i][2] = 0 # resets amount filled
    

    # Adds up all costs accrude (by different i ranges) and adds the cost of currently opened bin too
    total_cost = 0
    for bin in bins:
        total_cost += bin[0]
        if bin[1] > 0:
            total_cost += 1 + max(bin[2] - g, 0)*b
    
    return total_cost