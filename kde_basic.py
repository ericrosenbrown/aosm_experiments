import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


m1 = np.random.normal(scale=0.5, size=1000)
m2 = np.random.normal(scale=0.5, size=1000)

x, y = m1, m2

print(x.shape)
print(y.shape)

xx, yy, zz = kde2D(x, y, 1.0)

plt.pcolormesh(xx, yy, zz)
plt.scatter(x, y, s=2, facecolor='white')
plt.show()