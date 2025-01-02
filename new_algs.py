import numpy as np
import knapsack

def weight(item, i, g, b):
    return 1/i + max(0, (item - (g/i)) * b)

def create_items(g, b, tau, imax, nmax):
    items= np.zeros((2, nmax * imax))
    for i in range(1, imax+1):
        min, max = ((g+tau)/(i+1),(g+tau)/(i))
        steps = (max - min)/nmax
        for n in range(nmax):
           items[0][(i-1) * nmax + n] = (np.round(np.floor((min+steps*n) * 100) /100, 2))
           items[1][(i-1) * nmax + n] = (np.round(np.ceil(weight(min+steps*(n+1), i, g, b) * 100) /100, 2))

    return items

def dp_solver(items, max_size):
    return knapsack.knapsack(size=items[0], weight=items[1]).solve(max_size)

# UB on CR of GHAR(t) -- the imax and nmax parameters make the CR tighter, approaching the true CR from above
def ghar_CR_calc(g, b, tau, imax=10, nmax=5):
    items = create_items(g, b, tau, imax, nmax)
    round_up_max = np.round(np.ceil(g * 100) /100, 2)
    cr, chosen_items = dp_solver(items, max_size=round_up_max)

    return cr, chosen_items

# calculates the n function from the paper -- this is the integer, n, such that tau in [n-2/n+2 g , n-1/n+3 g] 
def n(tau, g):
    n = 0
    while n < 100:
        if tau >= g*(n-2)/(n+2) and tau <= g*(n-1)/(n+3):
            return n
        n += 1
    
    return 100


# Calculates the LOWER bound on the competitive ratio of the Thresholded almost anyfit alg for particular g,b,tau
def AAF_LB_CR_calc(g, b, tau):
    case1 = 2 / (1 + tau * b)
    case2 = g*(1 + tau*b) / (g + tau)
    n_val = n(tau, g)
    case3 = (1 + n_val + b*max(((n_val*(g-tau)/2) - g), 0)) / n_val

    return max(case1, case2, case3)

# Calculates the UPPER bound on the competitive ratio of the Thresholded almost anyfit alg for particular g,b,tau
def AAF_UB_CR_calc(g, b, tau):
    case1 = (2*g) / (g+tau)
    case2 = g*(1 + tau*b) / (g + tau)
    
    return max(case1, case2)


