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

# SETUP AND CREATE ALL NECESSARY STORING MATRICES #

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

#############################################################

# RUN AN EMPRIRICAL TEST AND STORE ALL COSTS #

# for i,g in enumerate(tqdm(gs)):
#     for j,b in enumerate(bs):
#         # example = np.random.rand(100) 
#         example = np.random.zipf(1.4267, 100) 
#         example = example / max(example)
#         # if g*b >= 1:
#         #     example *= g
        
#         if g*b > 1:

#             GHAR_store[i,j] = Ghar(example, g, b, 1/b, 100)
#             worst_fit_store[i,j] = worst_fit(example, g, b, threshold=g+1/b)

#             # GHAR_CR_store[i,j], _ = ghar_CR_calc(g, b, 1/b)
#             # AAF_LB_CR_store[i,j] = AAF_LB_CR_calc(g, b, 1/b)

#         gb_store[i,j] = g*b >= 1
      

# PLOTTING #

# g_b_heatmap(GHAR_store, "GHAR cost", number=number, gmax=gmax, bmax=bmax, gs=gs)
# g_b_heatmap(worst_fit_store, "WF cost", number=number, gmax=gmax, bmax=bmax, gs=gs)
# g_b_heatmap(GHAR_store - worst_fit_store, "GHAR - WF cost", number=number, gmax=gmax, bmax=bmax, gs=gs)
# g_b_heatmap(worst_fit_store >= GHAR_store, "WF worse than GHAR", number=number, gmax=gmax, bmax=bmax, gs=gs)
# quit()

# g_b_heatmap(GHAR_CR_store, "GHAR CR for tau = 1/b", number=number, bmax=bmax, gs=gs)
# g_b_heatmap(AAF_LB_CR_store, "AAF LB CR for tau = 1/b", number=number, bmax=bmax, gs=gs)
# g_b_heatmap(GHAR_CR_store - AAF_LB_CR_store, "GHAR CR - AAF LB CR for tau = 1/b", number=number, bmax=bmax, gs=gs)


# SEPARATE EMPRIRICAL TEST #

example = np.random.rand(1000) 
# example = np.random.zipf(1.4267, 1000) 
# example = example / max(example)
g_test = 0.6
b_test = 5/2
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


# PLOTTING FOR COMPARISON OF GHAR AND WF #

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
        


