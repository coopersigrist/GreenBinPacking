import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def g_b_heatmap(arr, title, number, gmax, bmax, gs):
    ax = sns.heatmap(arr, linewidth=0, xticklabels=[], yticklabels=[])
    plt.title(title)
    plt.xlabel("Value of B")
    plt.ylabel("Value of G")

    plot_bs = np.round(np.arange(0, bmax* 11/10, bmax/10),1)
    plot_gs = np.round(np.arange(gmax, -gmax/10, -gmax/10),1)

    axis_ticks = np.arange(0, (number+1)*11/10, (number+1)/10)

    plt.xticks(axis_ticks)
    ax.set_xticklabels(plot_bs)

    plt.yticks(axis_ticks)
    ax.set_yticklabels(plot_gs)

    # plt.plot((number/(bmax * gs)), number - number*gs - 1, label="G=1/B" )
    lineval = (number - number*number/(bmax*np.arange(number+2) + 0.0001))
    plt.plot(np.arange(number+2), lineval+1, label="G=1/B" )

    plt.legend()
    plt.show()