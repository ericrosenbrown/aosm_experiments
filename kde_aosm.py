import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import pickle

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min()-1:x.max()+1:xbins, 
                      y.min()-1:y.max()+1:ybins]
    print(xx,yy)
    xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T
    print(xy_sample)
    xy_train  = np.vstack([x, y]).T

    #epanechnikov
    #exponential
    kde_skl = KernelDensity(bandwidth=bandwidth, kernel = "gaussian", **kwargs,)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape), kde_skl

data_dir = "./aosm_data"
action, obj = ["ToggleObjectOn", "Toaster"]
#aosm_file = action + "_" + obj + "_directlook.p"
aosm_file = action + "_" + obj + "_relativeangles.p"

#action, obj = ["PickupObject", "Potato"]
#aosm_file = action + "_" + obj + "_angles.p"

aosm_data = pickle.load(open(data_dir+"/"+aosm_file,"rb"))

X_success = aosm_data["success"]
X_failure= aosm_data["failure"]



#x, y = zip(*X_success)
x, y, angles = zip(*X_success)
x = np.array(x)
y = np.array(y)


xx, yy, zz, kde_skl = kde2D(x, y, 0.2)


plt.pcolormesh(xx, yy, zz)
plt.scatter(x, y, s=2, facecolor='white')

"""
kde_samples = kde_skl.sample(100)
sx = kde_samples[:,0]
sy = kde_samples[:,1]
plt.scatter(sx, sy, s=2, facecolor='red')
"""
plt.show()