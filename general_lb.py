# Used for Fig 1, 2
#   Fig 1: General LB vs AAF/HAR LB (BG <= 1)
#   Fig 2: General LB vs TAAF/GHAR LB (BG > 1)
#   

import numpy as np
import matplotlib.pyplot as plt

#
# BG <= 1
#
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

def plot_1(gs, nB):
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


#
# BG > 1
#
def taaf_lb(gb, smooth = False):
    if gb < 3.63746:
        return 2*gb/(gb+1)
    else:
        return (14*gb+1)/(8*gb+4)

def general_lb(gb):
    if gb < 1.5:
        return 1
    elif gb < 3:
        return (4.5*(1+gb/3))/(4+1.5*(1+gb/3))
    elif gb < 4:
        return 9/7
    # elif gb < 5:
    #     return (2+2*gb/5)/(2+gb/5)
    elif gb < 13.2:
        return 4/3
    else:
        return 1.5
    
def plot_gen_lb(maxBG):
    gbs = np.linspace(1, maxBG, 500)

    taaf = []
    general = []
    for gb in gbs:
        taaf.append(taaf_lb(gb))
        general.append(general_lb(gb))

    gbs = np.linspace(1, maxBG, 500)
    plt.plot(gbs, taaf, label = "TAAF/GHAR LB")
    plt.plot(gbs, general, label = "General LB")
    plt.xlabel('\u03B2G')
    plt.ylabel('Competitive Ratio')
    plt.legend()
    plt.show()


# 
# Function calls
#

#Fig 1a: G <= 0.5 
plot_1([0.1,0.2,0.3,0.4,0.5], 1000)
#Fig 1b: G > 0.5
plot_1([0.6,0.7,0.8], 1000)

# Fig 2a: zoom in
plot_gen_lb(20)
# Fig 2b: zoom out
plot_gen_lb(100)
