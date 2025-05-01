#   Used for Fig 3, 4, 5
#   Fig 3: CR vs B for WF
#   Fig 4: CR vs B for AAF/HAR
#   Fig 5: CR vs GB for TNF, TWF, TAAF/GHAR
import numpy as np
import matplotlib.pyplot as plt

#
# BG <= 1
#

#TODO



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
        sqrt = nf_helper(gb)
        return (3*gb-2+sqrt)/(gb+sqrt)

def tnf_helper(gb):
    return (5*(gb**2)-8*gb+4)**0.5

def plot_5():
    gbs = np.linspace(1,100, 1000)

    taaf = []
    tnf = []
    twf = []

    for gb in gbs:
        taaf.append(taaf_cr(gb))
        tnf.append(tnf_cr(gb))
        twf.append(twf_cr(gb))

    gbs = np.linspace(1,100, 1000)
    plt.plot(gbs, taaf, label="TAAF & GHAR", linestyle = "solid", color = "tab:green")
    plt.plot(gbs, tnf, label ="TNF", linestyle = "-", color="tab:grey")
    plt.plot(gbs, twf, label ="TWF", linestyle = "--", color="tab:red")
    plt.legend()
    plt.xlabel("\u03B2G", fontsize="12")
    plt.ylabel("Optimal Competitive Ratio", fontsize="12")
    plt.show()

#
# Function calls
#

# Fig 5
plot_5()