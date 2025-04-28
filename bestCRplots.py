import numpy as np
import matplotlib.pyplot as plt


def taaf_cr(gb):
    if gb <= 3.637:
        return 2*gb/(gb+1)
    else:
        return (14*gb+1)/(8*gb+4)


def wf_cr(gb):
    return 2*gb/(gb+1)

def nf_cr(gb):
    if gb >= 2:
        return gb*(1+gb**0.5)/(gb+gb**0.5)
    else:
        sqrt = nf_helper(gb)
        return (3*gb-2+sqrt)/(gb+sqrt)

def nf_helper(gb):
    return (5*(gb**2)-8*gb+4)**0.5

def draw():
    gbs = np.linspace(1,100, 1000)

    taaf = []
    nf = []
    wf = []

    for gb in gbs:
        taaf.append(taaf_cr(gb))
        nf.append(nf_cr(gb))
        wf.append(wf_cr(gb))

    gbs = np.linspace(1,100, 1000)
    plt.plot(gbs, taaf, label="TAAF & GHAR", linestyle = "solid", color = "tab:green")
    plt.plot(gbs, nf, label ="TNF", linestyle = "-", color="tab:grey")
    plt.plot(gbs, wf, label ="TWF", linestyle = "--", color="tab:red")
    plt.legend()
    plt.xlabel("\u03B2G", fontsize="12")
    plt.ylabel("Optimal Competitive Ratio", fontsize="12")
    plt.show()


draw()