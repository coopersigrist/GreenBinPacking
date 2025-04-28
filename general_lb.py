import numpy as np
import matplotlib.pyplot as plt


# def cr_partial(a, k, B, G):
#     pos = G/(3*k*a)-G if G/(3*k*a)-G > 0 else 0
#     return 3*a*k*(1+B*pos)

# def lower_bound_a(G, nB, k = 20):
#     a_s = np.linspace(G/(3*k),1/k, 10)

#     for a in a_s:
#         bs = np.linspace(1/G, 10, nB)
#         res = []

#         for b in bs:
#             res.append(cr_partial(a,k,b,G))
        
#         plt.plot(bs, res, label=f"a:{a}")
    
#     plt.legend()
#     plt.show()

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
    

def plot_gen_lb():
    gbs = np.linspace(1, 20, 500)

    taaf = []
    general = []
    for gb in gbs:
        taaf.append(taaf_lb(gb))
        general.append(general_lb(gb))


    plt.plot(gbs, taaf, label = "TAAF LB")
    plt.plot(gbs, general, label = "General LB")
    plt.legend()
    plt.show()

plot_gen_lb()


