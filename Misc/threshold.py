import numpy as np
from matplotlib.pyplot import figure, show


def f1(g,b,t):
    return 2/(1+t*b)

def f2(g,b,t):
    return 1+(g-t)*(1+b*t)/(2*(g+t))

# For Each G in [0.01, ... 0.99], B = 1.02...10
#   Find the lowest point of LB in range t in [1/gB^2, 1/B]
#      store in an array
# plot 2d image

gs = (x * 0.01 for x in range(1,100))
results = []
for g in gs:

    bs = (x * 0.1 for x in range(10,100))
    bResults = []

    for b in bs:
        if g*b > 1:

            ts = np.linspace(1/(g*b*b), 1/b, 100, True)
            resultsArray = []

            for t in ts:
                resultsArray.append((t, np.maximum(f1(g,b,t), f2(g,b,t))))

            currentMinValue = 3
            currentMinT = 1

            for res in resultsArray:
                if res[1] < currentMinValue:
                    currentMinValue = res[1]
                    currentMinT = res[0]
            
            if currentMinT == 1/b:
                bResults.append(2)

            else:
                bResults.append(1)

        else:
            bResults.append(0)

    results.append(bResults)

print(len(results))
print(len(results[0]))

fig = figure()
ax = fig.add_subplot()

ys = np.linspace(0.01, .99, 99, True)
xs = np.linspace(1, 9.9, 90, True)

curveX = np.linspace(3.5,9.9,100)
curveY = 3.42/curveX
pc = ax.pcolormesh(xs, ys, results)
ax.plot(curveX, curveY, color = "red")
ax.set_xlabel("B")
ax.set_ylabel("G")
fig.colorbar(pc)
show()

        


        

