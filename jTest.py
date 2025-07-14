import numpy as np
import matplotlib.pyplot as plt

def draw(G, B):

    caps = np.linspace(G+1/(G*(B**2)), G+1/B, 500)
    items = np.linspace(.00001, G+1/(G*(B**2)), 500)

    res = []

    for i in items:
        tmp = []
        for c in caps:
            tmp.append((c//i)*i)    
        res.append(sum(tmp)/len(tmp))

    plt.plot(items, res)
    plt.show()

#draw(0.5,8)

def midpoint(G,B):
    return (B*G+1)/(2*B*B*G)

def draw2(i, G, B):
    caps = np.linspace(G+1/(G*(B**2)), G+1/B, 500)
    res = []

    for c in caps:
        res.append((c//i)*i/c)


    plt.plot(caps,res)
    plt.show()

def cr_helper(t,G,B):
    x1 = G+1/(G*(B**2))
    x2 = G+1/B
    return (2*(G+t-x1)/(x2-x1)+(x2-G-t)*(1+t*B)/(x2-x1)) / (1+t*B)

#print(cr_helper(.125, 0.5,8))

def draw3(G,B):
    x1 = G+1/(G*(B**2))
    x2 = G+1/B
    ts = np.linspace(1/(G*(B**2)), 1/B, 500)


    res = []
    for t in ts:
        res.append(cr_helper(t,G,B))
    
    plt.plot(ts, res)
    plt.show()
    
draw3(0.5,8)

def avg_fill(i,G,B):
    x1 = G+1/(G*(B**2))
    x2 = G+1/B

    p1 = 1/(2*(1/B - 1/(G*(B**2))))
    p2 = 2*(i**2)/x1 - x1/2 + x2 - 4*(i**2)/x2

    return p1*p2


def avg_fill2(i,G,B):
    x1 = G+1/(G*(B**2))
    x2 = G+1/B

    p1 = 1/(2*(1/B - 1/(G*(B**2))))
    p2 = 6*(i**2)/x1 - 2*x1/3 + x2 - 9*(i**2)/x2

    return p1*p2

#print(avg_fill2(.2083,0.5,8))

def avg_fill3(G,B):
    x1 = G+1/(G*(B**2))
    x2 = G+1/B

    # p1 = 1/(2*(1/B - 1/(G*(B**2))))
    # p2 = 2*(x2**2)/(3*x1) - 2*x1/3
    p1 = (x2**2)-(x1**2)
    p2 = 3*x1*(x2-x1)
    return p1/p2

#print(avg_fill3(0.5,8))

def derivative(G,B, i):
    x1 = G+1/(G*(B**2))
    x2 = G+1/B
    items = np.linspace(.00001, G+1/(G*(B**2)), 500)


    return 4*i*(1/x1 - 1/x2) - 1

def draw_der(G,B):
    items = np.linspace(.00001, G+1/(G*(B**2)), 500)
    res = []
    for i in items:
        res.append(derivative(i,G,B))

    plt.plot(items, res)
    plt.show()

def draw_af(G,B):
    x1 = G+1/(G*(B**2))
    x2 = G+1/B
    items = np.linspace((x1/2), x2, 500)
    res = []
    for i in items:
        res.append(avg_fill(i,G,B))

    plt.plot(items, res)
    plt.show()

#draw_af(.5,8)