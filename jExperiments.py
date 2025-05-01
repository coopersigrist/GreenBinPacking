# 
# This file covers all simulations and Fig 7 to determine best empirical tau
#

import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, solve, nsolve
import pickle
from algs import *
import os

#
#   Utility Functions
#
def read_from_file(filepath, n):
    with open(filepath, 'r') as f:
        numbers = []
        count = 0
        for line in f:
            count+=1
            line = line.strip()
            numbers.append(int(line)/100)
            if count > n:
                return numbers

    return numbers

def tauHat(g,b, guess = 0):
    x = Symbol('x', positive=True)
    cubic = -(b**2)*(x**3)+(x**2)*((b**2)*g)+x*((4*b*g)-3)-g
    # solutions = solve(cubic, x)
    solutions = nsolve(cubic, x, guess, prec=6)
    return float(solutions)

# Default tauhat_2.pickle for empirical_large tests
def populate_tauHat_file2(gs = [0.7,0.75,0.8, 0.85, 0.9, 0.95], bMax = 10, nB = 30):
    data = {}

    for g in gs:
        bs = np.linspace(1.005/g,bMax/g,40)
        for b in bs:
            guess = 1/(2*b)
            data[(g,b)]=tauHat(g,b, guess)

    with open('./Data/tauHat2.pickle', 'wb') as write:
        pickle.dump(data, write, protocol=-1)
    
## Tauhat file
def populate_tauHat_file():
    bs = [2,4,8,16]
    data = {}

    for b in bs:
        gs = np.linspace(1.005/b,1,15)
        for g in gs:
            guess = 1/(2*b)
            if g == 0.18808035714285715: 
                guess = 0
            data[(g,b)]=tauHat(g,b, guess)

    with open('./Data/tauHat.pickle', 'wb') as write:
        pickle.dump(data, write, protocol=-1)
    
def read_tauHat_pickle(ending=""):
    with open(f'./Data/tauHat{ending}.pickle', 'rb') as read:
        return pickle.load(read)

# Optimal Cost
def simpleCost(S, L,  b, g):
    return (S/L)*(1+b*(L-g)) if L>g else (S/L)




### Empirical BG <= 1
# Fix B.
# Vary G from [0,1/B]
# Plot performance of WF,NF,FF,BF,HAR
def empirical_small(b, nG = 15, nS= 5000, source="Unif", filename=""):
    gs = np.linspace(0,1/b, nG)

    if source == "Unif":
        simulatedData = np.random.rand(nS)
    elif source == "GI":
        simulatedData = read_from_file("/Users/jacksonbibbens/Documents/UMass/GBP/GreenBinPacking/Data/GI.txt", nS)
    elif source == "Weibull":
        simulatedData = read_from_file("/Users/jacksonbibbens/Documents/UMass/GBP/GreenBinPacking/Data/weibull.txt", nS)
    

    nfCost = []
    wfCost = []
    ffCost = []
    bfCost = []
    harCost = []

    for g in gs:
        optLB = opt_s(simulatedData,g,b)
        nfCost.append(next_fit(simulatedData,g,b)/optLB)
        wfCost.append(worst_fit(simulatedData,g,b)/optLB)
        ffCost.append(first_fit(simulatedData,g,b)/optLB)
        bfCost.append(best_fit(simulatedData,g,b)/optLB)
        harCost.append(Ghar(simulatedData,g,b,1-g,10)/optLB)

    gs = b*np.linspace(0,1/b, nG)

    plt.plot(gs, nfCost, label = "NF", marker="o", color="tab:grey")
    plt.plot(gs, wfCost, label = "WF", marker="s", color="tab:red")
    plt.plot(gs, ffCost, label = "FF", marker="D", color="tab:green")
    plt.plot(gs, bfCost, label = "BF", marker="P", color="tab:blue")
    plt.plot(gs, harCost, label = "Har", marker="v", color="tab:purple")
    plt.legend(fontsize= "11")
    plt.xlabel("\u03B2G", fontsize= "12")
    plt.ylabel("Performance Ratio", fontsize= "12")
    plt.subplots_adjust(left = 0.13, right = 0.96, top = 0.95, bottom = 0.1)
 
    if filename == "" :
        plt.show()
    else:
        plt.savefig(f"./JSB_Plots/{source}_GB<1/{filename}", dpi=300)
    
    plt.clf()


### Empirical best tau
# Fix G,B and vary T
def empirical_tau(b,g, tMax, nT, nS):
    ts = np.linspace(0,tMax,nT)

    simulatedData = np.random.rand(nS)

    nfCost = []
    wfCost = []
    ffCost = []
    bfCost = []
    harCost = []

    for t in ts:
        nfCost.append(next_fit(simulatedData, g, b, g+t))
        wfCost.append(worst_fit(simulatedData, g, b, g+t))
        ffCost.append(first_fit(simulatedData, g, b, g+t))
        bfCost.append(best_fit(simulatedData, g, b, g+t))
        harCost.append(Ghar(simulatedData, g, b, t, 10))

    ts = np.linspace(0,tMax,nT)

    tHat = tauHat(g,b)
    plt.axvline(x=1/b, color = "black", ls='-', lw = 0.5)
    plt.axvline(x=1/(2*b), color = "black", ls='-', lw = 0.5)
    plt.axvline(x=tHat, color = "black", ls='-', lw = 0.5)

    plt.plot(ts, nfCost, label = "TNF", linestyle = "-", color="tab:grey")
    plt.plot(ts, wfCost, label = "TWF", linestyle = "--", color="tab:red")
    plt.plot(ts, ffCost, label = "TFF", linestyle = ":", color="tab:green")
    plt.plot(ts, bfCost, label = "TBF", linestyle = "-.", color="tab:blue")
    plt.plot(ts, harCost, label = "Ghar", linestyle = (0, (3, 1, 1, 1, 1, 1)), color="tab:purple")
    
    plt.text(1/b+.001, plt.ylim()[1]*.98, "1/\u03B2")
    plt.text(1/(2*b)+.001, plt.ylim()[1]*.98, "1/2\u03B2")
    plt.text(0, plt.ylim()[1]*.98, r'$\hat{\tau}$')

    plt.legend(fontsize= "11")
    plt.xlabel("\u03C4", fontsize="12")
    plt.ylabel("Cost", fontsize="12")
    plt.subplots_adjust(left = 0.13, right = 0.96, top = 0.95, bottom = 0.1)
    plt.show()
    

### Empirical BG > 1
## GB > 1: Fix G and vary B from 1/G to 10/G
## if sorted, then input will be sorted in increasing size
## if ff_tauhat, then TFF + TBF will use threshold tauhat, else 0. Tauhat requires tauHat2.pickle
## if filename != "", then save image to that location without preview 
def empirical_large(g, bMax = 10, nB=30, nS=3000, source = "Unif", ff_tauhat = True, sorted = True, filename=""):
    bs = np.linspace(1.005/g,bMax/g, nB)

    if source == "Unif":
        simulatedData = np.random.uniform(0, 1, nS)
    elif source == "GI":
        simulatedData = read_from_file("./Data/GI.txt", nS)
    elif source == "Weibull":
        simulatedData = read_from_file("./Data/weibull.txt", nS)
    
    tauHatData = read_tauHat_pickle("2")

    if sorted:
        simulatedData.sort()

    tnfCost = []
    twfCost = []
    tffCost = []
    tbfCost = []
    gharCost = []

    for b in bs:
        optLB = opt_s(simulatedData,g,b)

        #TNF and GHAR get 1/b
        tau = 1/b
        tnfCost.append(next_fit(simulatedData,g,b, g+tau)/optLB)
        gharCost.append(Ghar(simulatedData,g,b,tau,10)/optLB)

        #WF gets 1/2b
        tau = 1/(2*b)
        twfCost.append(worst_fit(simulatedData,g,b, g+tau)/optLB)

        ## FF BF get tauHat
        if tauHat:
            tau = tauHatData[(g,b)]
        else:
            tau = 0
        tffCost.append(first_fit(simulatedData,g,b, g+tau)/optLB)
        tbfCost.append(best_fit(simulatedData,g,b, g+tau)/optLB)

    bs = g*np.linspace(1.005/g,bMax/g, nB)


    plt.plot(bs, tnfCost, label = "NF", marker="o", color="tab:grey")
    plt.plot(bs, twfCost, label = "WF", marker="s", color="tab:red")
    plt.plot(bs, tffCost, label = "FF", marker="D", color="tab:green")
    plt.plot(bs, tbfCost, label = "BF", marker="P", color="tab:blue")
    plt.plot(bs, gharCost, label = "Har", marker="v", color="tab:purple")
    
    plt.legend(fontsize="11")
    plt.xlabel("\u03B2G", fontsize="12")
    plt.ylabel("Performance Ratio", fontsize="12")
    plt.subplots_adjust(left = 0.13, right = 0.96, top = 0.95, bottom = 0.1)
 
    if filename == "" :
        plt.show()
    else:
        plt.savefig(f"./Simulation_Plots/FixG/{source}_GB>1/{filename}", dpi=300)
    
    plt.clf()



#
# Function calls
#

#Empirical tau selection
#    G = 0.5, test B = 5, 10, 20
empirical_tau(b=5, g=0.5, tMax=0.35, nT=1000,nS=3000)
empirical_tau(b=10, g=0.5, tMax=0.35, nT=1000,nS=3000)
empirical_tau(b=20, g=0.5, tMax=0.35, nT=1000,nS=3000)

# Empirical BG <= 1
#    B = 1, 1.5, 2, 4
empirical_small(b=4, filename="bar")

# Empirical BG > 1
#     Not going to list all combos here, theres a lot
# v1: B range from 1.005/g to 10, use tauhat, n3000
# v2: B range from 1.005/g to 10/g, use tauhat(?), n 3000
# v3: B range from 1.005/g to 10/g, use 0, n2000
# v4: B range from 1.005/g to 10/g, use tauhat, n2000
# v5: B range from 1.005/g to 10/g, use tauhat, n2000, Small items
# v6: B range from 1.005/g to 10/g, use 0, n2000, Small items

# Depending on the parameters you choose, you may need to regenerate tauHat2.pickle file
# populate_tauHat_file2()
empirical_large(g = 0.7, filename="foo")




