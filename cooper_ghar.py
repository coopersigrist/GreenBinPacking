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

def ghar_CR_calc(g, b, tau, imax=10, nmax=5):
    items = create_items(g, b, tau, imax, nmax)
    round_up_max = np.round(np.ceil(g * 100) /100, 2)
    cr, chosen_items = dp_solver(items, max_size=round_up_max)

    return cr, chosen_items
