"""
Taken from Mallasto's notebook on WGPs.

https://colab.research.google.com/drive/1gCpSOoZ6odPmDQxYwki4koYqciaolRQw#scrollTo=Kx5ruxXOipeB
"""

import numpy as np


def plot_ellipsoid(ax, pos, covariance_matrix, scale=0.025):
    U, s, rot = np.linalg.svd(covariance_matrix)
    smax = np.max(s)
    max_ind = np.argmax(s)
    s = scale * s / smax  # The scaling is just for nicer plots

    c = np.abs(rot[max_ind])
    # c = c / np.sum(c)

    u = np.linspace(0.0, 2.0 * np.pi, 50)
    v = np.linspace(0.0, np.pi, 50)
    x = s[0] * np.outer(np.cos(u), np.sin(v))
    y = s[1] * np.outer(np.sin(u), np.sin(v))
    z = s[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rot) + pos
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=c)
