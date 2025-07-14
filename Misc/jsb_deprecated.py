import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, solve, nsolve
import pickle
from algs import *


# Test 1
# Fix G (suggested {0.3,0.5,0,7})
# Vary B from [0,1/G]
# Plot performance of WF,NF,AAF,HAR(7)
def test1(g):
    bs = np.linspace(0,1/g, 10)
    simulatedData = np.random.rand(10)

    nfCost = []
    wfCost = []
    ffCost = []
    harCost = []

    for b in bs:
        nfCost.append(next_fit(simulatedData,g,b))
        wfCost.append(worst_fit(simulatedData,g,b))
        ffCost.append(first_fit(simulatedData,g,b))
        harCost.append(Har(simulatedData,g,b)[0])

    bs = np.linspace(0,1/g, 10)
    plt.plot(bs, nfCost)
    plt.plot(bs, wfCost)
    plt.plot(bs, ffCost)
    plt.plot(bs, harCost)
    plt.show()



def simpleCost(S, L,  b, g):
    return (S/L)*(1+b*(L-g)) if L>g else (S/L)

### GB > 1: fix B and vary G from 1/b to 1
def test3(b, nG = 15, nS = 5000, source="Unif", filename=""):
    gs = np.linspace(1.005/b,1, nG)

    if source == "Unif":
        simulatedData = np.random.rand(nS)
    elif source == "GI":
        simulatedData = read_from_file("/Users/jacksonbibbens/Documents/UMass/GBP/GreenBinPacking/Data/GI.txt", nS)
    elif source == "Weibull":
        simulatedData = read_from_file("/Users/jacksonbibbens/Documents/UMass/GBP/GreenBinPacking/Data/weibull.txt", nS)
    
    tauHatData = read_tauHat_pickle()

    nfCost = []
    wfCost = []
    ffCost = []
    bfCost = []
    harCost = []

    for g in gs:
        simulatedData = np.random.uniform(0, g, nS)
        optLB = opt_s(simulatedData,g,b)

        #NF and GHAR get 1/b
        tau = 1/b
        nfCost.append(next_fit(simulatedData,g,b, g+tau)/optLB)
        harCost.append(Ghar(simulatedData,g,b,tau,10)/optLB)

        #WF gets 1/2b
        tau = 1/(2*b)
        wfCost.append(worst_fit(simulatedData,g,b, g+tau)/optLB)

        ## FF BF get tauHat
        tau = tauHatData[(g,b)]
        # tau = 0
        ffCost.append(first_fit(simulatedData,g,b, g+tau)/optLB)
        bfCost.append(best_fit(simulatedData,g,b, g+tau)/optLB)

    gs = b*np.linspace(1.005/b,1, nG)

    plt.plot(gs, nfCost, label = "NF", marker="o", color="tab:grey")
    plt.plot(gs, wfCost, label = "WF", marker="s", color="tab:red")
    plt.plot(gs, ffCost, label = "FF", marker="D", color="tab:green")
    plt.plot(gs, bfCost, label = "BF", marker="P", color="tab:blue")
    plt.plot(gs, harCost, label = "Har", marker="v", color="tab:purple")
    
    plt.legend(fontsize="11")
    plt.xlabel("\u03B2G", fontsize="12")
    plt.ylabel("Performance Ratio", fontsize="12")
    plt.subplots_adjust(left = 0.13, right = 0.96, top = 0.95, bottom = 0.1)
 
    if filename == "" :
        plt.show()
    else:
        plt.savefig(f"./JSB_Plots/{source}_GB>1/{filename}", dpi=300)
    
    plt.clf()




### Investigating kick up GB<=1 
def test4(b, nG, S=100000):
    gs = np.linspace(0.5,1/b, nG)

    fill75 = []
    fill85 = []
    fill90 = []
    fill95 = []

    for g in gs:
        opt_LB = simpleCost(S,1,b,g)
        fill75.append(simpleCost(S,0.75,b,g)/opt_LB)
        fill85.append(simpleCost(S,0.85,b,g)/opt_LB)
        fill90.append(simpleCost(S,0.90,b,g)/opt_LB)
        fill95.append(simpleCost(S,0.95,b,g)/opt_LB)

    gs = b*np.linspace(0.5,1/b, nG)

    plt.plot(gs, fill75, label = "0.75")
    plt.plot(gs, fill85, label = "0.85")
    plt.plot(gs, fill90, label = "0.90")
    plt.plot(gs, fill95, label = "0.95")
    plt.legend()
    plt.xlabel("\u03B2G")
    plt.ylabel("Performance Ratio")
    plt.title(f"\u03B2 = {b}")
    plt.show()


### investigating hump GB>1
def test6(b, nG, S= 2000):
    gs = np.linspace(1.005/b,1, nG)

    fill75 = []
    fill85 = []
    fill90 = []
    fill95 = []

    for g in gs:
        thresh = g+1/(2*b)
        opt_LB = S/g
        fill75.append(simpleCost(S,0.75*thresh,b,g)/opt_LB)
        fill85.append(simpleCost(S,0.85*thresh,b,g)/opt_LB)
        fill90.append(simpleCost(S,0.90*thresh,b,g)/opt_LB)
        fill95.append(simpleCost(S,0.95*thresh,b,g)/opt_LB)

    gs = b*np.linspace(1.005/b,1, nG)

    plt.plot(gs, fill75, label = "0.75")
    plt.plot(gs, fill85, label = "0.85")
    plt.plot(gs, fill90, label = "0.90")
    plt.plot(gs, fill95, label = "0.95")
    plt.legend()
    plt.xlabel("\u03B2G")
    plt.ylabel("Performance Ratio")
    plt.title(f"\u03B2 = {b}")
    plt.show()
    



#cost on entire input
def gbgtone_cost(a, B, G):
    pos = 1/2-G if G < 1/2 else 0

    # Better to pack 1/2+e with 1/2-e
    if 1+B*(1-G) < 2*(1+B*pos):
        return a*(1+B*(1-G))+2*(1-a)*(1+B*pos)
    # Better to pack 1/2+e separately
    else:
        return (1-a)*(1+B*(1-G))+(2*a)*(1+B*pos)

# Cost when we only pack first set of items
def gbgtone_cost_2(a, B, G):
    pos = 1/2-G if G < 1/2 else 0
    return (1-a)*(1+B*(1-G))+(2*a-1)*(1+B*pos)  

# Opt cost: works for both first set and entire input, just divide by 2 for the first set
def gbgtone_opt(B,G):
    pos = 1/2-G if G < 1/2 else 0
    return min(1+B*(1-G), 2*(1+B*pos))

def lower_bound_2(G, nB=20):
    #a_s = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    a_s = [0.6, 2/3, 0.7,0.75,1]
    colors = ["red","blue","green","grey","purple","orange"]
    c= 0
    for a in a_s:
        bs = np.linspace(1/G, 10, nB)
        res = []
        res2 = []
        for b in bs:
            opt = gbgtone_opt(b,G)
            res.append(gbgtone_cost(a,b,G)/opt)
            res2.append(2*gbgtone_cost_2(a,b,G)/opt)

        plt.plot(bs, res, label= f'a:{a}', color = colors[c])
        plt.plot(bs, res2, label= f'a:{a} (Partial)', color = colors[c], linestyle = "--")
        c +=1 

    plt.legend()
    plt.show()            

def lower_bound_deprecated(r, nG = 30):
    bs = [1, 2, 4, 8]

    res = {}
    for b in bs:
        gs = np.linspace(0,1/b, nG)
        res[b] = []
        for g in gs:
            res[b].append(calc_lb(r,b, g))

    g1 = np.linspace(0,1, nG)
    g2 = np.linspace(0,1/2, nG)
    g4 = np.linspace(0,1/4, nG)
    g8 = np.linspace(0,1/8, nG)

    plt.plot(g1, res[1], label = "\u03B2 = 1", color="tab:grey")
    plt.plot(g2, res[2], label = "\u03B2 = 2", color="tab:red")
    plt.plot(g4, res[4], label = "\u03B2 = 4", color="tab:blue")
    plt.plot(g8, res[8], label = "\u03B2 = 8", color="tab:green")
    plt.legend(fontsize = "11")
    plt.xlabel("G", fontsize="12")
    plt.ylabel("Online Lower Bound", fontsize="12")
    plt.subplots_adjust(left = 0.13, right = 0.96, top = 0.95, bottom = 0.1)
    plt.show()
  

