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


# Test 2 
# Fix B.
# Vary G from [0,1/B]
# Plot performance of WF,NF,FF,HAR
def test2(b, nG = 15, nS= 5000, source="Unif", filename=""):
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

        # optCost.append(opt_s(simulatedData,g,b))
        # nfCost.append(next_fit(simulatedData,g,b))
        # wfCost.append(worst_fit(simulatedData,g,b)[0])
        # ffCost.append(first_fit(simulatedData,g,b))
        # harCost.append(Ghar(simulatedData,g,b,1-g,10))

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


def simpleCost(S, L,  b, g):
    return (S/L)*(1+b*(L-g)) if L>g else (S/L)

def tauHat(g,b, guess = 0):
    x = Symbol('x', positive=True)
    cubic = -(b**2)*(x**3)+(x**2)*((b**2)*g)+x*((4*b*g)-3)-g
    # solutions = solve(cubic, x)
    solutions = nsolve(cubic, x, guess, prec=6)
    return float(solutions)

def populate_tauHat_file2(bMax = 10):
    gs = [0.7,0.75,0.8, 0.85, 0.9, 0.95]
    data = {}

    for g in gs:
        bs = np.linspace(1.005/g,bMax/g,40)
        for b in bs:
            guess = 1/(2*b)
            data[(g,b)]=tauHat(g,b, guess)

    with open('/Users/jacksonbibbens/Documents/UMass/GBP/GreenBinPacking/Data/tauHat2.pickle', 'wb') as write:
        pickle.dump(data, write, protocol=-1)
    

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

    with open('/Users/jacksonbibbens/Documents/UMass/GBP/GreenBinPacking/Data/tauHat.pickle', 'wb') as write:
        pickle.dump(data, write, protocol=-1)
    
def read_tauHat_pickle(ending=""):
    with open(f'/Users/jacksonbibbens/Documents/UMass/GBP/GreenBinPacking/Data/tauHat{ending}.pickle', 'rb') as read:
        return pickle.load(read)

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


### Fix G,B and vary T
def test5(b,g, tMax, nT, nS):
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
    

    


### GB > 1: Fix G and vary B from 1/G to 10/G
### if sorted, then input will be sorted in increasing size
# v1; B range from 1.005/g to 10, use tauhat, n3000
# v2: B range from 1.005/g to 10/g, use tauhat(?), n 3000
# v3: B range from 1.005/g to 10/g, use 0, n2000
# v4: B range from 1.005/g to 10/g, use tauhat, n2000
# v5: B range from 1.005/g to 10/g, use tauhat, n2000, Small items
# v6: B range from 1.005/g to 10/g, use 0, n2000, Small items
def test7(g, bMax = 10, nB=30, nS=3000, source = "Unif", sorted = True, filename=""):
    bs = np.linspace(1.005/g,bMax/g, nB)

    if source == "Unif":
        simulatedData = np.random.uniform(0, 1, nS)
    elif source == "GI":
        simulatedData = read_from_file("/Users/jacksonbibbens/Documents/UMass/GBP/GreenBinPacking/Data/GI.txt", nS)
    elif source == "Weibull":
        simulatedData = read_from_file("/Users/jacksonbibbens/Documents/UMass/GBP/GreenBinPacking/Data/weibull.txt", nS)
    
    tauHatData = read_tauHat_pickle("2")

    if sorted:
        simulatedData.sort()

    nfCost = []
    wfCost = []
    ffCost = []
    bfCost = []
    harCost = []

    for b in bs:
        #simulatedData = np.random.uniform(0, 1, nS)
        optLB = opt_s(simulatedData,g,b)

        #NF and GHAR get 1/b
        # tau = 1/b
        # nfCost.append(next_fit(simulatedData,g,b, g+tau)/optLB)
        # harCost.append(Ghar(simulatedData,g,b,tau,10)/optLB)

        #WF gets 1/2b
        tau = 1/(2*b)
        wfCost.append(worst_fit(simulatedData,g,b, g+tau)[0]/optLB)

        ## FF BF get tauHat
        tau = tauHatData[(g,b)]
        # tau = 0
        ffCost.append(first_fit(simulatedData,g,b, g+tau)/optLB)
        bbbbffff = best_fit(simulatedData,g,b, g+tau)[0]
        bfCost.append(bbbbffff/optLB)

    bs = g*np.linspace(1.005/g,bMax/g, nB)


    # plt.plot(bs, nfCost, label = "NF", marker="o", color="tab:grey")
    plt.plot(bs, wfCost, label = "WF", marker="s", color="tab:red")
    plt.plot(bs, ffCost, label = "FF", marker="D", color="tab:green")
    plt.plot(bs, bfCost, label = "BF", marker="P", color="tab:blue")
    # plt.plot(bs, harCost, label = "Har", marker="v", color="tab:purple")
    
    plt.legend(fontsize="11")
    plt.xlabel("\u03B2G", fontsize="12")
    plt.ylabel("Performance Ratio", fontsize="12")
    plt.subplots_adjust(left = 0.13, right = 0.96, top = 0.95, bottom = 0.1)
 
    if filename == "" :
        plt.show()
    else:
        plt.savefig(f"./JSB_Plots/FixG/{source}_GB>1/{filename}", dpi=300)
    
    plt.clf()


def online_lb(r ,B, G):
    pos = 1-r*G if (1-r*G) > 0 else 0
    return (r+B*pos)/(1+B*(1-G))

def aaf_lb(B,G):
    if G <= 1/2:
        return (71/42 + B*(1805/1806 - 71*G/42))/(1+B-B*G)
    elif G <= 2/3:
        return (71/42 + B*(451/903 - 29*G/42))/(1+B-B*G)
    elif G <= 6/7:
        return (71/42 + B*(50/301 - 4*G/21))/(1+B-B*G)
    elif G <= 42/43:
         return (71/42 + B*(1/43 - G/42))/(1+B-B*G)
    else:
        return (71/42)/(1+B-B*G)


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


def lower_bound(gs, nB):
    colors = ["red","blue","green","grey","purple"]
    c = 0

    for g in gs:
        bs = np.linspace(1, 1/g, nB)

        online_res = []
        aaf_res = []

        for b in bs:
            online_res.append(online_lb(1.54037, b, g))
            aaf_res.append(aaf_lb(b,g))


        plt.plot(bs, online_res, label=f"Online: G = {g}", color = colors[c])
        plt.plot(bs, aaf_res, label=f"Alg: G = {g}", linestyle = '--', color = colors[c])
        c += 1

    plt.legend()
    plt.xlabel("\u03B2")
    plt.ylabel("Lower Bound")
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



# Run the tests
#
# 
# test1(0.3)
### INTERPRETATION
# NF is worst. WF and FF do the same, 
# both equal NF when GB = 1. Har easily wins


# lower_bound([0.51, 0.6, 0.65, 0.7 ], 20)
# lower_bound([0.1,0.2, 0.3, 0.4, 0.5], 20)


# lower_bound_2(0.75)
# print(gbgtone_cost(1, 1/.45, 0.45))
# print(gbgtone_opt(1/.45,0.5))


#test5(20, 0.5, 0.35, 25, 5000)

# simulatedData = np.random.rand(2500)
# # first_fit(simulatedData,0.5,2, 0.5+data)

populate_tauHat_file2(bMax = 20)

# test7(g = 0.7, sorted=False, nS = 2000,filename="g0dot7_v3")
# test7(g = 0.75, sorted=False, nS = 2000,filename="g0dot75_v3")
# test7(g = 0.8, sorted=False, nS = 2000, filename="g0dot8_v3")
# test7(g = 0.85, sorted=False, nS = 2000, filename="g0dot85_v3")
# test7(g = 0.95, sorted=False, nS = 2000, filename="g0dot95_v3")


# test7(g = 0.7, sorted=False, source = "Weibull", nS = 2000,filename="g0dot7_4")
# test7(g = 0.75, sorted=False,  source = "Weibull", nS = 2000,filename="g0dot75_v4")
# test7(g = 0.8, sorted=False, source = "Weibull", nS = 2000, filename="g0dot8_v4")
# test7(g = 0.85, sorted=False, source = "Weibull", nS = 2000, filename="g0dot85_v4")
test7(g = 0.9, bMax = 20, nB = 40, sorted=False, nS = 2000)
# test7(g = 0.95, sorted=False, source = "Weibull", nS = 2000, filename="g0dot95_v4")

# test7(g = 0.7, sorted=True, source = "Weibull", nS = 2000,filename="g0dot7_sorted_v4")
# test7(g = 0.75, sorted=True,  source = "Weibull", nS = 2000,filename="g0dot75_sorted_v4")
# test7(g = 0.8, sorted=True, source = "Weibull", nS = 2000, filename="g0dot8_sorted_v4")
# test7(g = 0.85, sorted=True, source = "Weibull", nS = 2000, filename="g0dot85_sorted_v4")
# test7(g = 0.95, sorted=True, source = "Weibull", nS = 2000, filename="g0dot95_sorted_v4")

# test7(g = 0.95, nB=20, nS = 2000)
# simulatedData = np.random.rand(2000)
# bfCost, bfBins = best_fit(simulatedData, 0.6, 1.005/.6, 0)
# print(f"BfCost:{bfCost}")
# simulatedData.sort()
# bfCost, bfBins = best_fit(simulatedData, 0.6, 1.005/.6, 0)
# print(f"BfCost SORTED:{bfCost}")
# size = np.sum(simulatedData)
# wfCost, wfBins = worst_fit(simulatedData, 0.6, 5, 0.1)
# bfCost, bfBins = best_fit(simulatedData, 0.6, 5, 0)
# print(f"WfCost:{wfCost}")
# print(f"BfCost:{bfCost}")
# print(f"Total Size:{size}")
# print(f"WFBins:{len(wfBins)}")
# print(f"BFBins:{len(bfBins)}")
# print(f"OptLB:{opt_s(simulatedData, 0.6, 5)}")


# test4(b = 1, nG= 30)w

# bs = np.linspace(0,2, 10)
# simulatedData = np.random.rand(10)
# print(next_fit(simulatedData, 0.5, bs[6]))
# print(opt_s(simulatedData,0.3,2))
# print(Ghar(simulatedData, 0.3, 2, 0.7, 5))

# g = 0.46875
# b=16
# simData = np.random.rand(5000)
# print(np.sum(simData))
# ffCost, ffBins = first_fit(simData,g,b,g+1/b)
# print(ffCost)

# print(ffCost)
# print(len(ffBins))

# optCost = opt_s(simData,g,b)
# print(optCost)
# print(ffCost/optCost)

# plt.hist(ffBins, bins = 100)
# plt.show()
# print(b*np.linspace(1.005/b,1, 15))
# test6(16,20)


# V1: large items, tauHat (may be some issues here with negative tauHat)
# V2: large items, 0
# V3: small items, 0
# V4: small items, tauhat
