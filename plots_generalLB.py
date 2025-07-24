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

def plot_1(gs, nB, plot = True):
    colors = ["red","blue","green","grey","purple"]
    c = 0

    plt.figure(figsize=(6.4,4))

    online = []
    aaf = []

    for g in gs:
        bs = np.linspace(1, 1/g, nB)

        online_res = []
        aaf_res = []

        for b in bs:
            online_res.append(online_lb(1.54037, b, g))
            aaf_res.append(aaf_lb(b,g))

        online.append(online_res)
        aaf.append(aaf_res)

        if plot:
            plt.plot(bs, online_res, label=f"Online LB: G = {g}", color = colors[c])
            plt.plot(bs, aaf_res, label=f"Alg LB: G = {g}", linestyle = '--', color = colors[c])
            c += 1

    if plot:
        plt.legend(fontsize = "12")
        plt.xlabel("\u03B2", fontsize = "12")
        plt.ylabel("Lower Bound", fontsize = "12")
        plt.yticks(np.linspace(1.00, 1.45, 10, True))
        
        plt.subplots_adjust(left = 0.13, right = 0.98, top = 0.98, bottom = 0.13)
        plt.show()  
    else:
        return online, aaf


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
        #return (1.5) + (3-gb)/(2*(1-gb))
        return 3*(gb+1)/(gb+5)
    elif gb < 3:
        # return (4.5*(1+gb/3))/(4+1.5*(1+gb/3))
        return 3*(gb+3)/(gb+11)
    elif gb < 4:
        return 9/7
    # elif gb < 5:
    #     return (2+2*gb/5)/(2+gb/5)
    elif gb < 48:
        return 4/3
    else:
        return 1.5
    
def plot_gen_lb(maxBG, plot = True):
    gbs = np.linspace(1, maxBG, 500)

    taaf = []
    general = []
    for gb in gbs:
        taaf.append(taaf_lb(gb))
        general.append(general_lb(gb))

    if plot:
        gbs = np.linspace(1, maxBG, 500)
        plt.plot(gbs, taaf, label = "TAAF/GHAR LB")
        plt.plot(gbs, general, label = "General LB")
        plt.xlabel('\u03B2G')
        plt.ylabel('Competitive Ratio')
        plt.legend()
        plt.show()
    else:
        return general, taaf


def multiplot_LB(data):
    _, axs = plt.subplots(1, 4, figsize=(10, 3))
    legend_fontsize = "9"

    axs[0].plot(data[0][0][0], label =  "Online LB: G = 0.1", linestyle = (0,(1,1)), color = "tab:grey" )
    axs[0].plot(data[0][0][1], label =  "Online LB: G = 0.2", linestyle = (0,(2,1)), color = "tab:red" )
    axs[0].plot(data[0][0][2], label =  "Online LB: G = 0.3", linestyle = (0,(3,1)), color = "tab:blue" )
    axs[0].plot(data[0][0][3], label =  "Online LB: G = 0.4", linestyle = (0,(4,1)), color = "tab:green" )
    axs[0].plot(data[0][0][4], label =  "Online LB: G = 0.5", linestyle = (0,(5,1)), color = "tab:purple" )
    axs[0].plot(data[0][1][0], label =  "AAF LB: G = 0.1", color = "tab:grey" )
    axs[0].plot(data[0][1][1], label =  "AAF LB: G = 0.2", color = "tab:red" )
    axs[0].plot(data[0][1][2], label =  "AAF LB: G = 0.3", color = "tab:blue" )
    axs[0].plot(data[0][1][3], label =  "AAF LB: G = 0.4", color = "tab:green" )
    axs[0].plot(data[0][1][4], label =  "AAF LB: G = 0.5", color = "tab:purple" )
    axs[0].set_ylabel("Competitive Ratio", fontsize="12")
    axs[0].set_xlabel("\u03B2G",fontsize="12")
    axs[0].set_yticks((1.0,1.1, 1.2, 1.3, 1.4))
    axs[0].legend(fontsize = legend_fontsize)

    axs[1].plot(data[1][0][0], label =  "Online LB: G = 0.1", linestyle = (0,(1,1)), color = "tab:grey" )
    axs[1].plot(data[1][0][1], label =  "Online LB: G = 0.2", linestyle = (0,(2,1)), color = "tab:red" )
    axs[1].plot(data[1][0][2], label =  "Online LB: G = 0.3", linestyle = (0,(3,1)), color = "tab:blue" )
    axs[1].plot(data[1][1][0], label =  "AAF LB: G = 0.1", color = "tab:grey" )
    axs[1].plot(data[1][1][1], label =  "AAF LB: G = 0.2", color = "tab:red" )
    axs[1].plot(data[1][1][2], label =  "AAF LB: G = 0.3", color = "tab:blue" )
    axs[1].set_xlabel("\u03B2G", fontsize="12")
    axs[1].set_yticks((1.0,1.1, 1.2, 1.3, 1.4))
    axs[1].legend(fontsize = legend_fontsize)

    axs[2].plot(data[2][0], label =  "Online LB", linestyle = (0,(1,1)), color = "tab:grey")
    axs[2].plot(data[2][1], label =  "TAAF LB", color = "tab:red")
    axs[2].set_xlabel("\u03B2G", fontsize="12")
    axs[2].set_yticks((1.0, 1.25, 1.5, 1.75))
    axs[2].legend(fontsize = legend_fontsize)

    axs[3].plot(data[3][0], label =  "Online LB", linestyle = (0,(1,1)), color = "tab:grey")
    axs[3].plot(data[3][1], label =  "TAAF LB", color = "tab:red")
    axs[3].set_xlabel("\u03B2G", fontsize="12")
    axs[3].set_yticks((1.0, 1.25, 1.5, 1.75))
    axs[3].legend(fontsize = legend_fontsize)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.9])
    plt.savefig(f"./Plots/Simulation/multiplot_generalLB.png", dpi=300)
    plt.clf()

# 
# Function calls
#

#Fig 1a: G <= 0.5 
#plot_1([0.1,0.2,0.3,0.4,0.5], 1000)
#Fig 1b: G > 0.5
#plot_1([0.6,0.7,0.8], 1000)

# Fig 2a: zoom in
# plot_gen_lb(20)
# Fig 2b: zoom out
# plot_gen_lb(100)

data0 = plot_1([0.1,0.2,0.3,0.4,0.5], 1000, plot = False)
data1 = plot_1([0.6,0.7,0.8], 1000, plot = False)
data2 = plot_gen_lb(5,False)
data3 = plot_gen_lb(100,False)
multiplot_LB((data0,data1,data2,data3))