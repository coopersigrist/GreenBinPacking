import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import math
from matplotlib.colors import LinearSegmentedColormap
from algs import *
from new_algs import *
from utils import *

number = 120
bmax = 50
gmax = 1

gs = np.round(np.arange(gmax, -.001, -gmax/(number)), 4) 
bs = np.round(np.arange(0, bmax+0.01, bmax/number), 4) 


plot_bs = np.round(np.arange(0, bmax* 11/10, bmax/10),1)
plot_gs = np.round(np.arange(gmax, -gmax/10, -gmax/10),1)


axis_ticks = np.arange(0, (number)*11/10, (number)/10)


num_threshs = 100
threshs = np.round(np.arange(0, 1, 1/num_threshs), 4)

GHAR_store = np.zeros((number+1, number+1))
worst_fit_store = np.zeros((number+1, number+1))

GHAR_CR_store = np.ones((number+1, number+1))
AAF_LB_CR_store = np.ones((number+1, number+1)) 

gb_store = np.zeros((number+1, number+1)) 

best_fit_cases = [[],[]]
green_best_fit_cases = [[], []]

# for n in range(1,number):
#     for imax in range(1,number):
#         GHAR_CR_val = np.zeros((number,number))
#         for i,g in enumerate(tqdm(gs)):
#             for j,b in enumerate(bs):
#                 if g*b > 1:
#                     GHAR_CR_val[n,imax] = cooper_ghar.ghar_CR_calc(g,b,1/b,imax,n)[0]
#         GHAR_CR_store[n,imax] = np.mean(GHAR_CR_val)

# ax = sns.heatmap(GHAR_CR_store, linewidth=0)
# plt.title("Mean CR for n vs imax")
# plt.xlabel("Value of imax")
# # plt.xticks(axis_ticks)
# # ax.set_xticklabels(plot_bs)
# plt.ylabel("Value of n")
# # plt.yticks(axis_ticks)
# # ax.set_yticklabels(plot_gs)
# # plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()


costarr = []
gval =[]
bval = []
# example = np.random.rand(100) #sample on the same distr
# example = np.random.zipf(1.4267, 100) 
# example = example / max(example)

# for i,g in enumerate(tqdm(gs)):
#     for j,b in enumerate(bs):
#         # example = np.random.rand(100) 

#         cost = []
#         if g*b>1:
#             gval.append(g)
#             bval.append(b)
#             for imax in range(100):
#                     cost.append(Ghar(example,g,b,1/b,imax))
#             costarr.append(cost)

# for i,plots in enumerate(costarr):
#      plt.plot(plots,label="g/b ratio" + str(round(gval[i]/bval[i],3)))
# plt.title("GHAR Zipf - Cost vs imax")
# plt.ylabel("cost")
# plt.xlabel("imax")
# plt.legend()
# plt.show()

for i,g in enumerate(tqdm(gs)):
    for j,b in enumerate(bs):
        # example = np.random.rand(100) 
        example = np.random.zipf(1.4267, 100) 
        example = example / max(example)
        # if g*b >= 1:
        #     example *= g
        
        if g*b > 1:

            GHAR_store[i,j] = Ghar(example, g, b, 1/b, 100)
            worst_fit_store[i,j] = worst_fit(example, g, b, threshold=g+1/b)

            # GHAR_CR_store[i,j], _ = ghar_CR_calc(g, b, 1/b)
            # AAF_LB_CR_store[i,j] = AAF_LB_CR_calc(g, b, 1/b)

        gb_store[i,j] = g*b >= 1

    
        

g_b_heatmap(GHAR_store, "GHAR cost", number=number, gmax=gmax, bmax=bmax, gs=gs)
g_b_heatmap(worst_fit_store, "WF cost", number=number, gmax=gmax, bmax=bmax, gs=gs)
g_b_heatmap(GHAR_store - worst_fit_store, "GHAR - WF cost", number=number, gmax=gmax, bmax=bmax, gs=gs)
g_b_heatmap(worst_fit_store >= GHAR_store, "WF worse than GHAR", number=number, gmax=gmax, bmax=bmax, gs=gs)
quit()

# g_b_heatmap(GHAR_CR_store, "GHAR CR for tau = 1/b", number=number, bmax=bmax, gs=gs)
# g_b_heatmap(AAF_LB_CR_store, "AAF LB CR for tau = 1/b", number=number, bmax=bmax, gs=gs)
# g_b_heatmap(GHAR_CR_store - AAF_LB_CR_store, "GHAR CR - AAF LB CR for tau = 1/b", number=number, bmax=bmax, gs=gs)


example = np.random.rand(1000) 
g_test = 0.2
b_test = 15/2
wf_cost = []
ghar_crs = []
aaf_lb_crs = []
aaf_ub_crs = []
ghar_cost = []

test_range = np.arange(100*(1-g_test))/100

for tau in test_range:
    wf_cost.append(worst_fit(example, g_test, b_test, g_test + tau))
    ghar_cost.append(Ghar(example, g_test, b_test, tau))
    cr, choices = ghar_CR_calc(g_test, b_test, tau)
    ghar_crs.append(cr)
    aaf_lb_crs.append(AAF_LB_CR_calc(g_test, b_test, tau))
    aaf_ub_crs.append(AAF_UB_CR_calc(g_test, b_test, tau))


plt.plot(test_range, aaf_lb_crs, label="AAF LB by tau")
plt.plot(test_range, aaf_ub_crs, label="AAF UB by tau")
plt.plot(test_range, ghar_crs, label="best CR of GHAR for emirical sequence")
plt.vlines(1/b_test, min(ghar_crs+aaf_lb_crs+aaf_ub_crs), max(ghar_crs+aaf_lb_crs+aaf_ub_crs), colors='red', label="tau = 1/b")
plt.vlines(1/(g_test*b_test*b_test), min(ghar_crs+aaf_lb_crs+aaf_ub_crs), max(ghar_crs+aaf_lb_crs+aaf_ub_crs), colors='red', label="tau = 1/gb^2")
plt.legend()
plt.xlabel("Tau")
plt.ylabel("Competitive ratio")
plt.title("CR over tau with g="+str(g_test)+" and b="+str(b_test))
plt.show()

plt.plot(test_range, wf_cost, label="empirical WF cost based on tau choice")
plt.plot(test_range, ghar_cost, label="empirical GHAR cost based on tau choice")
plt.vlines(1/b_test, min(wf_cost), max(wf_cost), colors='red')
plt.vlines(1/(g_test*b_test*b_test), min(wf_cost), max(wf_cost), colors='red')
plt.title("Cost over tau with g="+str(g_test)+" and b="+str(b_test))
plt.xlabel("Tau")
plt.ylabel("Cost")
plt.legend()
plt.show()
        

        # opt_s_store[i,j] = opt_s(example, g, b)
        # gaaf_s_store[i,j] = gaaf_s(example, g, b)
        # # nf_s_store[i,j] = nf_s(example, g, b)
        # green_next_fit_store[i,j] = green_next_fit(example, g, b)
        # green_worst_fit_store[i,j] = green_worst_fit(example, g, b)
        # green_best_fit_store[i,j] = green_best_fit(example, g, b)
        # green_first_fit_store[i,j] = green_first_fit(example, g, b)
        # next_fit_store[i,j] = next_fit(example, g, b, threshold=g)
        # worst_fit_store[i,j] = worst_fit(example, g, b, threshold=g)
        # best_fit_store[i,j] = best_fit(example, g, b, threshold=g)
        # first_fit_store[i,j] = first_fit(example, g, b, threshold=g)


        # for k,thresh in enumerate(threshs):
        #     wf_k = worst_fit(example, g, b, threshold=thresh)
        #     if worst_fit_store[i,j] == 0 or wf_k < worst_fit_store[i,j]:
        #         worst_fit_store[i,j] = wf_k
        #         best_wf_thresh[i,j] = k/num_threshs
        #         diff_from_g[0,i,j] = g - best_wf_thresh[i,j]
            
        #     bf_k = best_fit(example, g, b, threshold=thresh)
        #     if best_fit_store[i,j] == 0 or bf_k < best_fit_store[i,j]:
        #         best_fit_store[i,j] = bf_k
        #         best_bf_thresh[i,j] = k/num_threshs
        #         diff_from_g[1,i,j] = g - best_bf_thresh[i,j]

        #     ff_k = first_fit(example, g, b, threshold=thresh)
        #     if first_fit_store[i,j] == 0 or ff_k < first_fit_store[i,j]:
        #         first_fit_store[i,j] = ff_k
        #         best_ff_thresh[i,j] = k/num_threshs
        #         diff_from_g[2,i,j] = g - best_ff_thresh[i,j]



        # green_costs = np.array([green_next_fit_store[i,j], green_worst_fit_store[i,j], green_best_fit_store[i,j], green_first_fit_store[i,j]])
        # trad_costs = np.array([next_fit_store[i,j], worst_fit_store[i,j], best_fit_store[i,j], first_fit_store[i,j]])
        # all_costs = np.array([green_next_fit_store[i,j], green_worst_fit_store[i,j], green_best_fit_store[i,j], green_first_fit_store[i,j], next_fit_store[i,j], worst_fit_store[i,j], best_fit_store[i,j], first_fit_store[i,j] ])

        # whose_best[i,j] = np.argmin(all_costs)
        # best_alg_store[i,j] = np.min(all_costs)
        # best_green_alg_store[i,j] = np.min(green_costs)
        # best_trad_alg_store[i,j] = np.min(trad_costs)

        # if best_fit_store[i,j] < green_best_fit_store[i,j]:
        #     best_fit_cases[0].append(b * g)
        #     best_fit_cases[1].append(best_fit_store[i,j])
        # else:
        #     green_best_fit_cases[0].append(b*g)
        #     green_best_fit_cases[1].append(green_best_fit_store[i,j])

        # alg_store[0, i*number + j] = green_next_fit_store[i,j]
        # alg_store[1, i*number + j] = green_worst_fit_store[i,j]
        # alg_store[2, i*number + j] = green_best_fit_store[i,j]
        # alg_store[3, i*number + j] = green_first_fit_store[i,j]
        # alg_store[4, i*number + j] = opt_s_store[i,j]
        # alg_store[5, i*number + j] = nf_s_store[i,j]
        # alg_store[6, i*number + j] = next_fit_store[i,j]
        # alg_store[7, i*number + j] = worst_fit_store[i,j]
        # alg_store[8, i*number + j] = best_fit_store[i,j]
        # alg_store[9, i*number + j] = first_fit_store[i,j]


        # gb[i*number + j] = g * b



    ## COST GRAPHS ##
    # plt.scatter(gb, alg_store[2], label="GBF")
    # plt.scatter(gb, alg_store[3], label="GFF")
    # plt.scatter(gb, alg_store[8], label="BF")
    # plt.scatter(gb, alg_store[9], label="FF")

    # plt.scatter(gb, alg_store[4], label="OPT(S)")

    # ax = sns.heatmap((best_alg_store), linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("Minimum costs between all basic (green and standard) algorithms")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
    # plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
    # plt.legend()
    # plt.show()
    # plt.show()

    ## B=1/g graphs ##

    # plt.title("Costs of each alg when g = 1/b - OPT")
    # plt.xlabel("Value of G")
    # plt.ylabel("Cost")
    # # plt.plot(gs.flatten(), opt_s_store[:,0], label="OPT(S)")
    # plt.plot(gs.flatten(), best_fit_store[:,0] - opt_s_store[:,0], label="BF")
    # plt.plot(gs.flatten(), green_best_fit_store[:,0] - opt_s_store[:,0], label="GBF")
    # plt.plot(gs.flatten(), worst_fit_store[:,0] - opt_s_store[:,0], label="WF")
    # plt.plot(gs.flatten(), green_worst_fit_store[:,0] - opt_s_store[:,0], label="GWF")
    # plt.plot(gs.flatten(), first_fit_store[:,0] - opt_s_store[:,0], label="FF")
    # plt.plot(gs.flatten(), green_first_fit_store[:,0] - opt_s_store[:,0], label="GFF")
    # plt.legend()
    # plt.show()

# ax = sns.heatmap(worst_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Empirical (Zipf) worst fit with threshold = G + 1/b ")
# plt.xlabel("Value of B")
# plt.xticks(axis_ticks)
# ax.set_xticklabels(plot_bs)
# plt.ylabel("Value of G")
# plt.yticks(axis_ticks)
# ax.set_yticklabels(plot_gs)
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()

# ax = sns.heatmap(GHAR_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Empirical (Zipf) GHAR with threshold = G + 1/b")
# plt.xlabel("Value of B")
# plt.xticks(axis_ticks)
# ax.set_xticklabels(plot_bs)
# plt.ylabel("Value of G")
# plt.yticks(axis_ticks)
# ax.set_yticklabels(plot_gs)
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()

# ax = sns.heatmap(GHAR_store - worst_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Empirical (Zipf) GHAR - Worst fit cost")
# plt.xlabel("Value of B")
# plt.xticks(axis_ticks)
# ax.set_xticklabels(plot_bs)
# plt.ylabel("Value of G")
# plt.yticks(axis_ticks)
# ax.set_yticklabels(plot_gs)
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()

# ax = sns.heatmap(GHAR_store > worst_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("G, b pairs with Empirical (Zipf) GHAR > Worst fit cost")
# plt.xlabel("Value of B")
# plt.xticks(axis_ticks)
# ax.set_xticklabels(plot_bs)
# plt.ylabel("Value of G")
# plt.yticks(axis_ticks)
# ax.set_yticklabels(plot_gs)
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()


# ax = sns.heatmap(HAR_store/opt_s_store, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("CR of HAR")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.legend()
# plt.show()

    ## RATIO GRAPHS ##

    # ax = sns.heatmap((best_alg_store / opt_s_store), linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("ratio of minimum cost alg to OPT")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
    # plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
    # plt.legend()
    # plt.show()
    # plt.show()

    # # plt.scatter(gb, alg_store[5], label="NF(S)")
    # # plt.scatter(gb, alg_store[0]/alg_store[4], label="GNF / OPT(S)")
    # # plt.scatter(gb, alg_store[1]/alg_store[4], label="GWF / OPT(S)")
    # plt.scatter(gb, alg_store[2]/alg_store[4], label="GBF / OPT(S)")
    # plt.scatter(gb, alg_store[3]/alg_store[4], label="GFF / OPT(S)")
    # # plt.scatter(gb, alg_store[6]/alg_store[4], label="NF / OPT(S)")
    # # plt.scatter(gb, alg_store[7]/alg_store[4], label="WF / OPT(S)")
    # plt.scatter(gb, alg_store[8]/alg_store[4], label="BF / OPT(S)")
    # plt.scatter(gb, alg_store[9]/alg_store[4], label="FF / OPT(S)")

    # # plt.scatter(gb, alg_store[4], label="OPT(S)")

    # plt.title("cost based on G * Beta")
    # plt.xlabel("G*Beta")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.show()

    ## BEST ALG GRAPHS ##

    # plt.scatter(best_fit_cases[0],best_fit_cases[1] , label="BF < GBF")
    # plt.scatter(green_best_fit_cases[0], green_best_fit_cases[1] , label="BF > GBF")
    # plt.title("cost based on G * Beta")
    # plt.xlabel("G*Beta")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.show()

    # ax = sns.heatmap(green_best_fit_store/best_fit_store, linewidth=0, xticklabels=[], yticklabels=[])
    # plt.title("Cost of GBF/BF for varied G and B")
    # plt.xlabel("Value of B")
    # plt.ylabel("Value of G")
    # # plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
    # # plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
    # # plt.legend()
    # plt.show()

# ax = sns.heatmap(best_bf_thresh, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best Threshold for BF")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(best_wf_thresh, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best Threshold for WF")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()
# ax = sns.heatmap(best_ff_thresh, linewidth=0, xticklabels=[], yticklabels=[])
# plt.title("Best Threshold for FF")
# plt.xlabel("Value of B")
# plt.ylabel("Value of G")
# plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
# plt.plot((number/(bmax * gs)),  number*gs + 1, label="G = 1 - 1/B" )
# plt.legend()
# plt.show()

