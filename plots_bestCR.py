#   Used for Fig 3, 4, 5
#   Fig 3: CR vs B for WF
#   Fig 4: CR vs B for AAF/HAR
#   Fig 5: CR vs GB for TNF, TWF, TAAF/GHAR
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import random


#
# BG <= 1
#

#TODO

def nf_cr(b,g):
    return (2+b*(1-g))/(b+1-b*g)

def wf_cr(b,g):
    pos = 1-2*g if g < 0.5 else 0
    return (2+b*pos)/(b + 1 - b * g)

def aaf_cr_lb(b,g):
    denom = 1+b*(1-g)

    if g <= 1/2:
        return (71/42+b*(1805/1806 - 71*g/42))/denom
    
    elif g <= 2/3:
        return (71/42+b*(451/903 - 29*g/42))/denom
    
    elif g <= 6/7:
        return (71/42+b*(50/301 - 8*g/42))/denom
    
    elif g < 42/43:
        return (71/42+b*(1/43 - g/42))/denom
    
    else:
        return (71/42)/denom
    

def aaf_cr_ub(b,g):
    denom = b+1-b*g

    if g <= 0.5:
        return (1.75+b*(1-1.75*g))/denom
    
    elif g <= 2/3 and b <= 0.25/(0.5-.75*g):
        return (1.75+b*(0.5-0.75*g))/denom
    
    elif g <= 2/3 and b > 0.25/(0.5-0.75*g):
        return (1.5+b*(1-g))/denom
    
    else:
        return 1.75

def plot_small(gs, fn, filename = ""):
    lines = ["-","--","-.",":", (0, (5, 5))]
    line_cycles = cycle(lines)
    colors = ['blue','orange','green','red','purple']
    colors_cycle = cycle(colors)
    for g in gs:
        bs = np.linspace(1, 1/g, 500)
        res = []
        for b in bs:
            res.append(fn(b,g))

        plt.plot(bs, res, label = f"G = {g}", linestyle = next(line_cycles), color = next(colors_cycle), linewidth = 2.5)
    
    plt.xlabel("\u03B2", fontsize="15")
    plt.ylabel("Competitive Ratio", fontsize="15")
    plt.legend(fontsize="14")
    plt.xticks(fontsize="13")
    plt.yticks(fontsize="13")
    plt.subplots_adjust(left = 0.12, right = 0.98, top = 0.98, bottom = 0.11)

    if filename:
        plt.savefig(f"./Plots/CompetitiveRatio/bestCR_gb<1/{filename}.png", dpi = 300)
    else:
        plt.show()

    plt.clf()

def plot_aaf(gs, filename=""):
    lines = ["--","-.",":", (0, (5, 5)), (0, (10,5))]
    line_cycles = cycle(lines)
    colors = ['blue','orange','green','red','purple']
    colors_cycle = cycle(colors)

    for g in gs:
        bs = np.linspace(1, 1/g, 500)
        lb = []
        ub = []
        for b in bs:
            lb.append(aaf_cr_lb(b,g))
            ub.append(aaf_cr_ub(b,g))

        c = next(colors_cycle)
        plt.plot(bs, lb, label = f"LB (G = {g})", linestyle = next(line_cycles), color = c, linewidth = 2.5)
        plt.plot(bs, ub, label = f"UB (G = {g})", color = c, linewidth = 2.5)

    plt.xlabel("\u03B2", fontsize="15")
    plt.ylabel("Competitive Ratio", fontsize="15")
    plt.legend(fontsize="14")
    plt.xticks(fontsize="13")
    plt.yticks(fontsize="13")
    plt.subplots_adjust(left = 0.12, right = 0.99, top = 0.99, bottom = 0.11)

    if filename:
        plt.savefig(f"./Plots/CompetitiveRatio/bestCR_gb<1/{filename}.png", dpi = 300)
    else:
        plt.show()

    plt.clf()

def multiplot(data, filename = ""):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    lines = ["--","-.",":", (0, (1, 1, 3, 1)), (0, (2,2))]
    colors = ['blue','orange','green','red','purple']
    lw = 1
    legend_fs = "7.5"
    label_fs = "11"

    ### Small G values
    
    #NF
    line_cycles = cycle(lines)
    colors_cycle = cycle(colors)
    for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
        bs = np.linspace(1, 1/g, 250)
        res = []
        for b in bs:
            res.append(nf_cr(b,g))
        axs[0][0].plot(bs, res, label=f"G = {g}", linestyle = next(line_cycles), color = next(colors_cycle), linewidth = lw)
        axs[0][0].legend(fontsize=legend_fs)
        axs[0][0].set_xlabel("\u03B2", fontsize=label_fs)
        axs[0][0].set_ylabel("Competitive Ratio", fontsize=label_fs)

    line_cycles = cycle(lines)
    colors_cycle = cycle(colors)
    for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
        bs = np.linspace(1, 1/g, 250)
        res = []
        for b in bs:
            res.append(wf_cr(b,g))
        axs[0][1].plot(bs, res, label=f"G = {g}", linestyle = next(line_cycles), color = next(colors_cycle), linewidth = lw)
        axs[0][1].legend(fontsize=legend_fs)
        axs[0][1].set_xlabel("\u03B2", fontsize=label_fs)
        axs[0][1].set_ylabel("Competitive Ratio", fontsize=label_fs)

    line_cycles = cycle(lines)
    colors_cycle = cycle(colors)
    for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
        bs = np.linspace(1, 1/g, 250)
        lb = []
        ub = []
        for b in bs:
            lb.append(aaf_cr_lb(b,g))
            ub.append(aaf_cr_ub(b,g))
        c = next(colors_cycle)
        axs[0][2].plot(bs, lb, label=f"LB: G = {g}", linestyle = next(line_cycles), color = c, linewidth = lw)
        axs[0][2].plot(bs, ub, label=f"UB: G = {g}", linestyle = "-", color = c, linewidth = lw)
        axs[0][2].legend(fontsize=legend_fs)
        axs[0][2].set_xlabel("\u03B2", fontsize=label_fs)
        axs[0][2].set_ylabel("Competitive Ratio", fontsize=label_fs)


    axs[0][0].sharey(axs[0][1])
    axs[0][1].sharey(axs[0][2])
    plt.setp(axs[0][1].get_yticklabels(), visible=False)
    plt.setp(axs[0][2].get_yticklabels(), visible=False)

    # Large G values
        
    line_cycles = cycle(lines)
    colors_cycle = cycle(colors)
    for g in [0.55, 0.6, 0.65]:
        bs = np.linspace(1, 1/g, 250)
        res = []
        for b in bs:
            res.append(nf_cr(b,g))
        axs[1][0].plot(bs, res, label=f"G = {g}", linestyle = next(line_cycles), color = next(colors_cycle), linewidth = lw)
        axs[1][0].legend(fontsize=legend_fs, loc=4)
        axs[1][0].set_xlabel("\u03B2", fontsize=label_fs)
        axs[1][0].set_ylabel("Competitive Ratio", fontsize=label_fs)

    line_cycles = cycle(lines)
    colors_cycle = cycle(colors)
    for g in [0.55, 0.6, 0.65]:
        bs = np.linspace(1, 1/g, 250)
        res = []
        for b in bs:
            res.append(wf_cr(b,g))
        axs[1][1].plot(bs, res, label=f"G = {g}", linestyle = next(line_cycles), color = next(colors_cycle), linewidth = lw)
        axs[1][1].legend(fontsize=legend_fs)
        axs[1][1].set_xlabel("\u03B2", fontsize=label_fs)
        axs[1][1].set_ylabel("Competitive Ratio", fontsize=label_fs)

    line_cycles = cycle(lines)
    colors_cycle = cycle(colors)
    for g in [0.55, 0.6, 0.65]:
        bs = np.linspace(1, 1/g, 250)
        lb = []
        ub = []
        for b in bs:
            lb.append(aaf_cr_lb(b,g))
            ub.append(aaf_cr_ub(b,g))
        c = next(colors_cycle)
        axs[1][2].plot(bs, lb, label=f"LB: G = {g}", linestyle = next(line_cycles), color = c, linewidth = lw)
        axs[1][2].plot(bs, ub, label=f"UB: G = {g}", linestyle = "-", color = c, linewidth = lw)
        axs[1][2].legend(fontsize=legend_fs)
        axs[1][2].set_xlabel("\u03B2", fontsize=label_fs)
        axs[1][2].set_ylabel("Competitive Ratio", fontsize=label_fs)

    axs[1][0].sharey(axs[1][1])
    axs[1][1].sharey(axs[1][2])
    plt.setp(axs[1][1].get_yticklabels(), visible=False)
    plt.setp(axs[1][2].get_yticklabels(), visible=False)

    plt.subplots_adjust(hspace= 0.28)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("./Plots/CompetitiveRatio/bestCR_gb<1/multiplot.png",dpi=300)
    plt.clf()


#
# BG > 1
#

def taaf_cr(gb):
    if gb <= 3.637:
        return 2*gb/(gb+1)
    else:
        return (14*gb+1)/(8*gb+4)

def twf_cr(gb):
    return 2*gb/(gb+1)

def tnf_cr(gb):
    if gb >= 2:
        return gb*(1+gb**0.5)/(gb+gb**0.5)
    else:
        sqrt = tnf_helper(gb)
        return (3*gb-2+sqrt)/(gb+sqrt)

def tnf_helper(gb):
    return (5*(gb**2)-8*gb+4)**0.5

def plot_5(filename=""):
    gbs = np.linspace(1,100, 1000)

    taaf = []
    tnf = []
    twf = []

    for gb in gbs:
        taaf.append(taaf_cr(gb))
        tnf.append(tnf_cr(gb))
        twf.append(twf_cr(gb))

    gbs = np.linspace(1,100, 1000)
    plt.plot(gbs, taaf, label="TAAF & GHAR", linestyle = "solid", color = "tab:green", linewidth=2)
    plt.plot(gbs, tnf, label ="TNF", linestyle = "-", color="tab:grey", linewidth=2)
    plt.plot(gbs, twf, label ="TWF", linestyle = "--", color="tab:red", linewidth=2)
    plt.legend()
    plt.xlabel("\u03B2G", fontsize="14")
    plt.ylabel("Competitive Ratio", fontsize="14")
    plt.legend(fontsize="13")
    plt.xticks(fontsize="12")
    plt.yticks(fontsize="12")
    
    if filename:
        plt.savefig(f"./Plots/CompetitiveRatio/{filename}.png", dpi = 300)
    else:
        plt.show()

#
# Function calls
#
# plot_small([0.1,0.2,0.3,0.4,0.5], nf_cr, 'nf_small_g')
# plot_small([0.6,0.7,0.8], nf_cr, 'nf_large_g')
# plot_small([0.1,0.2,0.3,0.4,0.5], wf_cr, 'wf_small_g')
# plot_small([0.6, 0.7, 0.8], wf_cr, 'wf_large_g')
# plot_aaf([0.1,0.2,0.3,0.4,0.5], "aaf_small_g")
# plot_aaf([0.5, 0.6, 0.65], "aaf_large_g")

multiplot("hi")

# Fig 5
# plot_5('bestCR_gb>1')