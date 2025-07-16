# 
# This file covers all simulations and Fig 7 to determine best empirical tau
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from sympy import Symbol, nsolve
import pickle
from algs import *
import os
import random

#
#   Utility Functions
#
def read_from_file(filepath, n, skip=0):
    with open(filepath, 'r') as f:
        numbers = []
        count = 0

        for line in f:
            if skip > 0:
                skip -= 1
                continue

            count+=1
            line = line.strip()
            numbers.append(float(line))

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
        bs = np.linspace(1.005/g,bMax/g,nB)
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
def empirical_small(b, nG = 30, nS= 3000, source="Unif", filename="", plot=True):
    gs = np.linspace(0,1/b, nG)

    if source == "Unif":
        simulatedData = np.random.rand(nS)
    elif source == "GI":
        skip = random.randint(0,1000)
        simulatedData = read_from_file("./Data/GI.txt", nS, skip)
    elif source == "Weibull":
        skip = random.randint(0, 10000)
        simulatedData = read_from_file("./Data/weibull.txt", nS, skip)
    

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

    if plot:
        gs = b*np.linspace(0,1/b, nG)

        plt.plot(gs, nfCost, label = "NF", marker="o", color="tab:grey")
        plt.plot(gs, wfCost, label = "WF", marker="s", color="tab:red")
        plt.plot(gs, ffCost, label = "FF", marker="D", color="tab:green")
        plt.plot(gs, bfCost, label = "BF", marker="P", color="tab:blue")
        plt.plot(gs, harCost, label = "Har", marker="v", color="tab:purple")
        plt.legend(fontsize= "11")
        plt.xlabel("\u03B2G", fontsize= "12")
        plt.ylabel("Empirical Competitive Ratio", fontsize="11")
        plt.subplots_adjust(left = 0.13, right = 0.96, top = 0.95, bottom = 0.1)
    
        if filename == "" :
            plt.show()
        else:
            plt.savefig(f"./Plots/Simulation/{source}_GB<1/{filename}", dpi=300)
        
        plt.clf()
        return
    
    else:
        return (nfCost, wfCost, ffCost, bfCost, harCost)


### Empirical best tau
# Fix G,B and vary T
def empirical_tau(b,g, tMax, nT, nS, filename="", plot=True, source="Unif"):
    ts = np.linspace(0,tMax,nT)

    if source == "Unif":
        simulatedData = np.random.rand(nS)
    elif source == "Weibull":
        simulatedData = read_from_file("./Data/weibull.txt", nS)

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


    if plot:
        ts = np.linspace(0,tMax,nT)
        tHat = tauHat(g,b)

        _, ax = plt.subplots()   
        plt.axvline(x=1/b, color = "black", ls='-', lw = 0.5)
        plt.axvline(x=1/(2*b), color = "black", ls='-', lw = 0.5)
        plt.axvline(x=tHat, color = "black", ls='-', lw = 0.5)

        plt.plot(ts, nfCost, label = "TNF$", linestyle = "-", color="tab:grey")
        plt.plot(ts, wfCost, label = "TWF", linestyle = "--", color="tab:red")
        plt.plot(ts, ffCost, label = "TFF", linestyle = ":", color="tab:green")
        plt.plot(ts, bfCost, label = "TBF", linestyle = "-.", color="tab:blue")
        plt.plot(ts, harCost, label = "Ghar", linestyle = (0, (3, 1, 1, 1, 1, 1)), color="tab:purple")
        
        plt.text(1/b+.004, plt.ylim()[1]*.98, "1/\u03B2", fontsize="13")
        plt.text(1/(2*b)+.001, plt.ylim()[1]*.98, r"$\frac{1}{2\beta}$", fontsize="15")
        plt.text(0, plt.ylim()[1]*.98, r'$\hat{\tau}$', fontsize="13")

        plt.legend(fontsize= "13")
        plt.xlabel("\u03C4", fontsize="15")
        plt.ylabel("Cost", fontsize="15")
        plt.xticks(fontsize="12")
        plt.xticks(fontsize="12")
        ax.yaxis.set_major_formatter(ticker.EngFormatter())
        plt.subplots_adjust(left = 0.13, right = 0.96, top = 0.95, bottom = 0.1)
        
        if filename == "" :
            plt.show()
        else:
            plt.savefig(f"./Plots/Simulation/tau_selection/{filename}", dpi=300)
        
        plt.clf()
        return

    else:
        return (nfCost, wfCost, ffCost, bfCost, harCost)
    

### Empirical BG > 1
## GB > 1: Fix G and vary B from 1/G to bMax/G
# nB: number of betas sampled
# nS: number of items per instance
# sorted: whether input will be sorted in increasing size
# smallItems: whether (unif distribution) input items will be set to max g
# smartOpt: Ensures items with size x > g attribute cost 1+B(x-G) to Opt
# tnf_t: threshold for TNF. {0,1,2} indicates 0, 1/b, sqrt{g/b}
# ghar_t: threshold for GHar. {0,1,2} indicates 0, 1/b, (1/b, 1/2b)
# twf_t: threshld for TWF. {0,1,2} indicates 0, 1/2b, 1/b
# taaf_t, thresholf for TBF and TFF. {0,1,2} indicates 0, tauhat, (1/b, 1/2b). Tauhat requires tauHat2.pickle
# if filename != "", then save image to that location without preview 
def empirical_large(g, bMax = 10, nB=30, nS=3000, source = "Unif", 
                    sorted = False, smartOpt=False, smallItems =False,
                    tnf_t = 1, ghar_t = 1, twf_t = 1, taaf_t = 1, 
                    plot = True,
                    filename=""):
    bs = np.linspace(1.005/g,bMax/g, nB)

    if source == "Unif":
        ub = g if smallItems else 1
        simulatedData = np.random.uniform(0, ub, nS)
    elif source == "GI":
        skip = random.randint(0,1000)
        simulatedData = read_from_file("./Data/GI.txt", nS, skip)
    elif source == "Weibull":
        skip = random.randint(0,10000)
        simulatedData = read_from_file("./Data/weibull.txt", nS, skip)
    
    tauHatData = read_tauHat_pickle("2")

    if sorted:
        simulatedData.sort()

    tnfCost = []
    twfCost = []
    tffCost = []
    tbfCost = []
    gharCost = []

    for b in bs:
        if smartOpt:
            optLB = opt_smart(simulatedData,g,b)
        else:
            optLB = opt_s(simulatedData,g,b)

        # print('b: '+str(b))
        # print("OptLB: "+str(optLB))

        #TNF
        tnf = [0, 1/b, math.sqrt(g/b)][tnf_t]
        tnfCost.append(next_fit(simulatedData,g,b, g+tnf)/optLB)

        #GHar
        if ghar_t == 2:
            ghar = 1/b if g*b < 3.63746 else 1/(2*b)
        else:
            ghar = [0, 1/b][ghar_t]
        gharCost.append(Ghar(simulatedData,g,b,ghar,10)/optLB)

        #TWF
        twf = [0, 1/(2*b),  1/b][twf_t]
        twfCost.append(worst_fit(simulatedData,g,b, g+twf)/optLB)

        ##TFF,TBF 
        if taaf_t == 2:
            taaf = 1/b if g*b < 3.63746 else 1/(2*b)
        elif taaf_t == 0:
            taaf = 0
        else:
            taaf =  tauHatData[(g,b)]

        tffCost.append(first_fit(simulatedData,g,b, g+taaf)/optLB)
        tbfCost.append(best_fit(simulatedData,g,b, g+taaf)/optLB)

    if plot: 
        bs = g*np.linspace(1.005/g,bMax/g, nB)

        plt.plot(bs, tnfCost, label = "NF",linestyle = "-", color="tab:grey")
        plt.plot(bs, twfCost, label = "WF", linestyle = (0, (5,3)), color="tab:red")
        plt.plot(bs, tffCost, label = "FF", linestyle = (3, (10,3,2,3)), color="tab:green")
        plt.plot(bs, tbfCost, label = "BF", linestyle = (0, (10,3,2,3,2,3)), color="tab:blue")
        plt.plot(bs, gharCost, label = "Har", linestyle = (0, (3,3)), color="tab:purple")
        
        plt.legend(fontsize="12")
        plt.xlabel("\u03B2G", fontsize="13")
        plt.ylabel("Empirical Competitive Ratio", fontsize="11")
        plt.subplots_adjust(left = 0.13, right = 0.96, top = 0.95, bottom = 0.1)
    
        if filename == "" :
            plt.show()
        else:
            #plt.savefig(f"./Plots/Simulation/FixG/{source}_GB>1/{filename}", dpi=300)
            plt.savefig(f"./Plots/Simulation/compare_threshold/{filename}", dpi=300)

        plt.clf()
        return 
    
    else:
        return (tnfCost, twfCost, tffCost, tbfCost, gharCost)


def multiplot_smallBG(data, gs, filename, share_Y = False):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3), sharey=share_Y)
    
    mpl.rcParams['lines.linewidth'] = 2
    # plot 1
    axs[0].plot(gs[0], data[0][0], label = "NextFit",linestyle = "-", color="tab:grey" )
    axs[0].plot(gs[0], data[0][1], label = "WorstFit",linestyle = (0, (5,2)), color="tab:red" )
    axs[0].plot(gs[0], data[0][2], label = "FirstFit",linestyle = (1, (8,2,3,2)), color="tab:green" )
    axs[0].plot(gs[0], data[0][3], label = "BestFit",linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
    axs[0].plot(gs[0], data[0][4], label = "Harmonic",linestyle = (0, (2,2)), color="tab:purple" )
    axs[0].set_xlabel("\u03B2G", fontsize="12")
    axs[0].set_ylabel("Empirical Competitive Ratio", fontsize="11")

    # plot 2-4
    for i in [1,2]:
        axs[i].plot(gs[i], data[i][0], linestyle = "-", color="tab:grey" )
        axs[i].plot(gs[i], data[i][1], linestyle = (0, (5,2)), color="tab:red" )
        axs[i].plot(gs[i], data[i][2], linestyle = (1, (8,2,3,2)), color="tab:green" )
        axs[i].plot(gs[i], data[i][3], linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
        axs[i].plot(gs[i], data[i][4], linestyle = (0, (2,2)), color="tab:purple" )
        axs[i].set_xlabel("\u03B2G", fontsize="12")
        axs[i].set_ylabel("Empirical Competitive Ratio", fontsize="11")

    handles, labels = [], []
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=5, fontsize="10")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"./Plots/Simulation/smallBG/{filename}", dpi=300)
    plt.clf()



def multiplot_compareThreshold(data, bs, filename, share_Y=False):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=share_Y)
    
    mpl.rcParams['lines.linewidth'] = 2
    # plot 1
    axs[0].plot(bs, data[0][0], label = r"NextFit$_\tau$",linestyle = "-", color="tab:grey" )
    axs[0].plot(bs, data[0][1], label = r"WorstFit$_\tau$",linestyle = (0, (5,2)), color="tab:red" )
    axs[0].plot(bs, data[0][2], label = r"FirstFit$_\tau$",linestyle = (1, (8,2,3,2)), color="tab:green" )
    axs[0].plot(bs, data[0][3], label = r"BestFit$_\tau$",linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
    axs[0].plot(bs, data[0][4], label = r"Harmonic$_\tau$",linestyle = (0, (2,2)), color="tab:purple" )
    axs[0].set_xlabel("\u03B2G", fontsize="12")
    axs[0].set_ylabel("Empirical Competitive Ratio", fontsize="11")
    #axs[0].set_yticks([1.0,1.1,1.2,1.3])

    # plot 2
    axs[1].plot(bs, data[1][0], linestyle = "-", color="tab:grey" )
    axs[1].plot(bs, data[1][1], linestyle = (0, (5,2)), color="tab:red" )
    axs[1].plot(bs, data[1][2], linestyle = (1, (8,2,3,2)), color="tab:green" )
    axs[1].plot(bs, data[1][3], linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
    axs[1].plot(bs, data[1][4], linestyle = (0, (2,2)), color="tab:purple" )
    axs[1].set_xlabel("\u03B2G", fontsize="12")
    axs[1].set_ylabel("Empirical Competitive Ratio", fontsize="11")
    #axs[1].set_yticks([1.0,1.1,1.2,1.3])

    # plot 3
    axs[2].plot(bs, data[2][0],linestyle = "-", color="tab:grey" )
    axs[2].plot(bs, data[2][1],linestyle = (0, (5,3)), color="tab:red" )
    axs[2].plot(bs, data[2][2],linestyle = (1, (8,2,3,2)), color="tab:green" )
    axs[2].plot(bs, data[2][3],linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
    axs[2].plot(bs, data[2][4],linestyle = (0, (2,2)), color="tab:purple" )
    axs[2].set_xlabel("\u03B2G", fontsize="12")
    axs[2].set_ylabel("Empirical Competitive Ratio", fontsize="11")
    #axs[2].set_yticks([1.0,1.2,1.4,1.6])

    handles, labels = [], []
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=5, fontsize="10")

    plt.tight_layout(rect=[0, 0.01, 1, 0.9])
    plt.savefig(f"./Plots/Simulation/compare_threshold/{filename}", dpi=300)
    plt.clf()

def multiplot_largeBG(data, bs, filename, share_Y = False):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=share_Y)
    
    mpl.rcParams['lines.linewidth'] = 2
    # plot 1
    axs[0].plot(bs[0], data[0][0], label = r"NextFit$_\tau$",linestyle = "-", color="tab:grey" )
    axs[0].plot(bs[0], data[0][1], label = r"WorstFit$_\tau$",linestyle = (0, (5,2)), color="tab:red" )
    axs[0].plot(bs[0], data[0][2], label = r"FirstFit$_\tau$",linestyle = (1, (8,2,3,2)), color="tab:green" )
    axs[0].plot(bs[0], data[0][3], label = r"BestFit$_\tau$",linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
    axs[0].plot(bs[0], data[0][4], label = r"Harmonic$_\tau$",linestyle = (0, (2,2)), color="tab:purple" )
    axs[0].set_xlabel("\u03B2G", fontsize="12")
    axs[0].set_ylabel("Empirical Competitive Ratio", fontsize="11")
    #axs[0].set_yticks([1.0,1.1,1.2,1.3])

    # plot 2
    axs[1].plot(bs[1], data[1][0], linestyle = "-", color="tab:grey" )
    axs[1].plot(bs[1], data[1][1], linestyle = (0, (5,2)), color="tab:red" )
    axs[1].plot(bs[1], data[1][2], linestyle = (1, (8,2,3,2)), color="tab:green" )
    axs[1].plot(bs[1], data[1][3], linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
    axs[1].plot(bs[1], data[1][4], linestyle = (0, (2,2)), color="tab:purple" )
    axs[1].set_xlabel("\u03B2G", fontsize="12")
    axs[1].set_ylabel("Empirical Competitive Ratio", fontsize="11")
    #axs[1].set_yticks([1.0,1.1,1.2,1.3])

    # plot 3
    axs[2].plot(bs[2], data[2][0],linestyle = "-", color="tab:grey" )
    axs[2].plot(bs[2], data[2][1],linestyle = (0, (5,3)), color="tab:red" )
    axs[2].plot(bs[2], data[2][2],linestyle = (1, (8,2,3,2)), color="tab:green" )
    axs[2].plot(bs[2], data[2][3],linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
    axs[2].plot(bs[2], data[2][4],linestyle = (0, (2,2)), color="tab:purple" )
    axs[2].set_xlabel("\u03B2G", fontsize="12")
    axs[2].set_ylabel("Empirical Competitive Ratio", fontsize="11")
    #axs[2].set_yticks([1.0,1.2,1.4,1.6])

    handles, labels = [], []
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=5, fontsize="10")

    plt.tight_layout(rect=[0, 0.01, 1, 0.9])
    plt.savefig(f"./Plots/Simulation/largeBG/{filename}", dpi=300)
    plt.clf()


def multiplot_varyThreshold(data, ts, b, tHat, filename=""):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    
    mpl.rcParams['lines.linewidth'] = 2
    # plot 1
    axs[0].plot(ts, data[0][0], label = r"NextFit$_\tau$",linestyle = "-", color="tab:grey" )
    axs[0].plot(ts, data[0][1], label = r"WorstFit$_\tau$",linestyle = (0, (5,2)), color="tab:red" )
    axs[0].plot(ts, data[0][2], label = r"FirstFit$_\tau$",linestyle = (1, (8,2,3,2)), color="tab:green" )
    axs[0].plot(ts, data[0][3], label = r"BestFit$_\tau$",linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
    axs[0].plot(ts, data[0][4], label = r"Harmonic$_\tau$",linestyle = (0, (2,2)), color="tab:purple" )
    axs[0].set_xlabel(r"$\tau$", fontsize="12")
    axs[0].set_ylabel("Cost", fontsize="12")
    axs[0].yaxis.set_major_formatter(ticker.EngFormatter())
    #axs[0].set_yticks([1.0,1.1,1.2,1.3])

    # plot 2
    axs[1].plot(ts, data[1][0], linestyle = "-", color="tab:grey" )
    axs[1].plot(ts, data[1][1], linestyle = (0, (5,2)), color="tab:red" )
    axs[1].plot(ts, data[1][2], linestyle = (1, (8,2,3,2)), color="tab:green" )
    axs[1].plot(ts, data[1][3], linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
    axs[1].plot(ts, data[1][4], linestyle = (0, (2,2)), color="tab:purple" )
    axs[1].set_xlabel(r"$\tau$", fontsize="12")
    axs[1].set_ylabel("Cost", fontsize="12")
    axs[1].yaxis.set_major_formatter(ticker.EngFormatter())
    #axs[1].set_yticks([1.0,1.1,1.2,1.3])

    # plot 3
    axs[2].plot(ts, data[2][0],linestyle = "-", color="tab:grey" )
    axs[2].plot(ts, data[2][1],linestyle = (0, (5,3)), color="tab:red" )
    axs[2].plot(ts, data[2][2],linestyle = (1, (8,2,3,2)), color="tab:green" )
    axs[2].plot(ts, data[2][3],linestyle = (0, (1,2,4,2,1,2)), color="tab:blue" )
    axs[2].plot(ts, data[2][4],linestyle = (0, (2,2)), color="tab:purple" )
    axs[2].set_xlabel(r"$\tau$", fontsize="12")
    axs[2].set_ylabel("Cost", fontsize="12")
    axs[2].yaxis.set_major_formatter(ticker.EngFormatter())
    #axs[2].set_yticks([1.0,1.2,1.4,1.6])

    for i in [0,1,2]:
            axs[i].axvline(x=1/b[i], color = "black", ls='-', lw = 0.5)
            axs[i].axvline(x=1/(2*b[i]), color = "black", ls='-', lw = 0.5)
            axs[i].axvline(x=tHat[i], color = "black", ls='-', lw = 0.5)
            axs[i].text(1/b[i]+.004, axs[i].get_ylim()[1]*(97-3*i)/100, r"$\frac{1}{\beta}$", fontsize="12")
            axs[i].text(1/(2*b[i])+.001, axs[i].get_ylim()[1]*(97-3*i)/100, r"$\frac{1}{2\beta}$", fontsize="12")
            axs[i].text(tHat[i]-0.02, axs[i].get_ylim()[1]*(97-3*i)/100, r'$\hat{\tau}$', fontsize="11")

    handles, labels = [], []
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=5, fontsize="10")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"./Plots/Simulation/tau_selection/{filename}.png", dpi=300)
    plt.clf()


def multiplot_hist(data):
    _, axs = plt.subplots(1, 3, figsize=(10, 3))

    axs[0].hist(data[0], bins = 100)
    axs[0].set_ylabel("Count", fontsize="12")
    axs[0].set_xlabel("Item Size",fontsize="12")

    axs[1].hist(data[1], bins = 100)
    axs[1].set_ylabel("Count", fontsize="12")
    axs[1].set_xlabel("Item Size", fontsize="12")

    axs[2].hist(data[2], bins = 80)
    axs[2].set_ylabel("Count", fontsize="12")
    axs[2].set_xlabel("Item Size", fontsize="12")
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.9])
    plt.savefig(f"./Plots/Simulation/multiplot_hist", dpi=300)
    plt.clf()


#
# Function calls
#

### HISTOGRAM
# data0 = np.random.random(5000)
# data1 = read_from_file("./Data/GI.txt", 5000)
# data2 = read_from_file("./Data/weibull.txt", 5000)
# multiplot_hist((data0,data1,data2))



### Empirical tau selection
#    G = 0.5, test B = 5, 10, 20

# data0 = empirical_tau(b=5, g=0.5, tMax=0.35, nT=150, nS=3000, plot=False, source = "Unif")
# data1 = empirical_tau(b=10, g=0.5, tMax=0.35, nT=150, nS=3000, plot=False, source = "Unif")
# data2 = empirical_tau(b=20, g=0.5, tMax=0.35, nT=150, nS=3000, plot=False, source = "Unif")
# ts = np.linspace(0,0.35,150)
# with open("./Plots/Simulation/tau_selection/unif.pickle", 'wb') as f:
#     pickle.dump((data0,data1,data2), f)
# multiplot_varyThreshold((data0,data1,data2), ts, b=[5,10,20], tHat=[0.06, 0.025,0.01])

# data0 = empirical_tau(b=5, g=0.5, tMax=0.35, nT=150, nS=3000, plot=False, source = "Weibull")
# data1 = empirical_tau(b=10, g=0.5, tMax=0.35, nT=150, nS=3000, plot=False, source = "Weibull")
# data2 = empirical_tau(b=20, g=0.5, tMax=0.35, nT=150, nS=3000, plot=False, source = "Weibull")
# ts = np.linspace(0,0.35,150)
# with open("./Plots/Simulation/tau_selection/weibull.pickle", 'rb') as f:
#     data = pickle.load( f)
# multiplot_varyThreshold(data, ts, b=[5,10,20], tHat=[0.06, 0.025,0.01], filename="multiplot_weibull")

### Empirical BG <= 1
#    B = 1, 1.5, 2, 4

# data0 = empirical_small(b=1, source = "GI", plot=False)
# data1 = empirical_small(b=1.5, source = "GI", plot=False)
# data3 = empirical_small(b=4, source = "GI", plot=False)
# gs = [d*np.linspace(0, 1/d, 30) for d in [1,1.5,4]]
# multiplot_smallBG((data0,data1,data3),gs, filename="multiplot_GI.png")

# data0 = empirical_small(b=1, source = "Unif", plot=False)
# data1 = empirical_small(b=1.5, source = "Unif", plot=False)
# data3 = empirical_small(b=4, source = "Unif", plot=False)
# gs = [d*np.linspace(0, 1/d, 30) for d in [1,1.5,4]]
# multiplot_smallBG((data0,data1,data3),gs, filename="multiplot_unif.png")

# data0 = empirical_small(b=1, source = "GI", plot=False)
# data1 = empirical_small(b=1.5, source = "GI", plot=False)
# data3 = empirical_small(b=4, source = "GI", plot=False)
# gs = [d*np.linspace(0, 1/d, 30) for d in [1,1.5,4]]
# multiplot_smallBG((data0,data1,data3),gs, filename="multiplot_weibull.png")

# data0 = empirical_small(b=1, source = "GI", plot=False)
# data1 = empirical_small(b=1.5, source = "GI", plot=False)
# data3 = empirical_small(b=4, source = "GI", plot=False)
# gs = [d*np.linspace(0, 1/d, 30) for d in [1,1.5,4]]
# multiplot_smallBG((data0,data1,data3),gs, filename="multiplot_GI2.png", share_Y=True)

# data0 = empirical_small(b=1, source = "Unif", plot=False)
# data1 = empirical_small(b=1.5, source = "Unif", plot=False)
# data3 = empirical_small(b=4, source = "Unif", plot=False)
# gs = [d*np.linspace(0, 1/d, 30) for d in [1,1.5,4]]
# multiplot_smallBG((data0,data1,data3),gs, filename="multiplot_unif2.png", share_Y=True)

# data0 = empirical_small(b=1, source = "GI", plot=False)
# data1 = empirical_small(b=1.5, source = "GI", plot=False)
# data3 = empirical_small(b=4, source = "GI", plot=False)
# gs = [d*np.linspace(0, 1/d, 30) for d in [1,1.5,4]]
# multiplot_smallBG((data0,data1,data3),gs, filename="multiplot_weibull2.png", share_Y=True)



# Empirical BG > 1
#     Not going to list all combos here, theres a lot
# v1: B range from 1.005/g to 10, use tauhat, n3000
# v2: B range from 1.005/g to 10/g, use tauhat(?), n 3000
# v3: B range from 1.005/g to 10/g, use 0, n2000
# v4: B range from 1.005/g to 10/g, use tauhat, n2000
# v5: B range from 1.005/g to 10/g, use tauhat, n2000, Small items
# v6: B range from 1.005/g to 10/g, use 0, n2000, Small items
# v7: B range from 1/005/g to 20/h, use tauhat, n3000
# v8: B range from 1/005/g to 20/h, use tauhat, n3000, "smartOpt", sorted
# v9: B range from 1/005/g to 20/h, use 0, n3000, "smartOpt", sorted


# Depending on the parameters you choose, you may need to regenerate tauHat2.pickle file
# populate_tauHat_file2(gs = [.7,.75,.8,.85,.9, .95], bMax=20 , nB=30)
# empirical_large(g = 0.7, bMax = 20, tauHat = True, smartOpt=True, filename="g0dot7_v8")
# empirical_large(g = 0.75, bMax = 20, tauHat = True, smartOpt=True, filename="g0dot75_v8")
# empirical_large(g = 0.8, bMax = 20, tauHat = True, smartOpt=True, filename="g0dot8_v8")
# empirical_large(g = 0.85, bMax = 20, tauHat = True, smartOpt=True, filename="g0dot85_v8")
# empirical_large(g = 0.9, bMax = 20, tauHat = True, smartOpt=True, filename="g0dot9_v8")
# empirical_large(g = 0.95, bMax = 20, tauHat = True, smartOpt=True, filename="g0dot95_v8")

### Compare tau = 0, empirical tau, theoretical tau
# data0 = empirical_large( g = 0.5, bMax = 20, tnf_t= 0, ghar_t= 0, twf_t = 0, taaf_t= 0, smartOpt=True, plot=False, source="Unif")
# data1 =  empirical_large( g = 0.5, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0, smartOpt=True, plot=False, source="Unif")
# data2 = empirical_large(g = 0.5, bMax = 20, tnf_t= 2, ghar_t= 2, twf_t = 2, taaf_t= 2, smartOpt=True, plot=False, source="Unif")
# bs = 0.5*np.linspace(1.005/0.5,20/0.5, 30)
# with open("./Plots/Simulation/compare_threshold/unif.pickle", 'r') as f:
#     data = pickle.load(f)
# multiplot_compareThreshold(data, bs, "multiplot_unif.png", share_Y=True)

# data0 = empirical_large( g = 0.5, bMax = 20, tnf_t= 0, ghar_t= 0, twf_t = 0, taaf_t= 0, smartOpt=True, plot=False, source="Weibull")
# data1 =  empirical_large( g = 0.5, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0, smartOpt=True, plot=False, source="Weibull")
# data2 = empirical_large(g = 0.5, bMax = 20, tnf_t= 2, ghar_t= 2, twf_t = 2, taaf_t= 2, smartOpt=True, plot=False, source="Weibull")
# bs = 0.5*np.linspace(1.005/0.5,20/0.5, 30)
# with open("./Plots/Simulation/compare_threshold/weibull.pickle", 'wb') as f:
#     pickle.dump((data0,data1,data2), f)
# multiplot_compareThreshold((data0,data1,data2), bs, "multiplot_weibull.png", share_Y=True)

### Sorted all 3 distributions
# populate_tauHat_file2(gs = [0.5, .75, .95], bMax=20 , nB=30)
data0 = empirical_large( g = 0.5, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0, source="Unif", sorted=True, smartOpt=True, plot=False)
data1 =  empirical_large( g = 0.5, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0, source="GI",sorted=True, smartOpt=True, plot=False)
data2 = empirical_large(g = 0.5, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0, source="Weibull",sorted=True, smartOpt=True, plot=False)
bs = 0.5*np.linspace(1.005/0.5,20/0.5, 30) 
with open("./Plots/Simulation/compare_threshold/sorted_variety.pickle", 'wb') as f:
    pickle.dump((data0,data1,data2), f)
multiplot_compareThreshold((data0,data1,data2), bs, filename="multiplot_sorted_variety.png", share_Y=True)

### Empirical plots by distribution

# data0 = empirical_large( g = 0.5, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0,source="Unif", smartOpt=True, plot=False)
# data1 =  empirical_large( g = 0.75, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t = 0,source="Unif", smartOpt=True, plot=False)
# data2 = empirical_large(g = 0.95, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0,source="Unif", smartOpt=True, plot=False)
# with open("./Plots/Simulation/largeBG/unif.pickle", 'wb') as f:
#     pickle.dump((data0,data1,data2), f)
# bs = [g*np.linspace(1.005/g,20/g, 30) for g in [0.5,.75,.95]]
# multiplot_largeBG((data0,data1,data2), bs, filename="multiplot_unif.png", share_Y=True)

# data0 = empirical_large( g = 0.5, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0,source="GI", smartOpt=True, plot=False)
# data1 =  empirical_large( g = 0.75, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t = 0,source="GI", smartOpt=True, plot=False)
# data2 = empirical_large(g = 0.95, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0,source="GI", smartOpt=True, plot=False)
# with open("./Plots/Simulation/largeBG/GI.pickle", 'wb') as f:
#     pickle.dump((data0,data1,data2), f)
# bs = [g*np.linspace(1.005/g,20/g, 30) for g in [0.5,.75,.95]]
# multiplot_largeBG((data0,data1,data2), bs, filename="multiplot_GI.png", share_Y=True)

# data0 = empirical_large( g = 0.5, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0, source="Weibull", smartOpt=True, plot=False)
# data1 =  empirical_large( g = 0.75, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t = 0, source="Weibull", smartOpt=True, plot=False)
# data2 = empirical_large(g = 0.95, bMax = 20, tnf_t= 1, ghar_t= 1, twf_t = 1, taaf_t= 0, source="Weibull", smartOpt=True, plot=False)
# with open("./Plots/Simulation/largeBG/weibull.pickle", 'wb') as f:
#     pickle.dump((data0,data1,data2), f)
# bs = [g*np.linspace(1.005/g,20/g, 30) for g in [0.5,.75,.95]]
# multiplot_largeBG((data0,data1,data2), bs, filename="multiplot_weibull.png", share_Y=True)

