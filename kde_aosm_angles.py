import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import pickle
import math


def rotate(origin, point, angle):
  """
  Rotate a point counterclockwise by a given angle around a given origin.

  The angle should be given in radians.
  """
  angle = math.radians(angle)
  ox, oy = origin
  px, py = point

  qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
  qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
  return qx, qy


def kde2D(x, y, a, fitted_angle, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min()-1:x.max()+1:xbins, 
                      y.min()-1:y.max()+1:ybins]

    
    fitted_angles = np.array([fitted_angle]*yy.ravel().shape[0])
    xy_sample = np.vstack([xx.ravel(), yy.ravel(), fitted_angles]).T

    xy_train  = np.vstack([x, y, a]).T

    #epanechnikov
    #exponential
    kde_skl = KernelDensity(bandwidth=bandwidth, kernel = "gaussian", **kwargs,)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape), kde_skl

viz_points = True
data_dir = "./aosm_data"
#action, obj = ["ToggleObjectOn", "Toaster"]
#aosm_file = action + "_" + obj + "_angles2.p"
#aosm_file = action + "_" + obj + "_relativeangles.p"

#action, obj = ["PickupObject", "Potato"]

#action, obj = ["PickupObject", "Mug"]
action, obj = ["ToggleObjectOn", "CoffeeMachine"]
#aosm_file = action + "_" + obj + "_angles.p"
#aosm_file = action + "_" + obj + "_relativeangles.p"
aosm_file = action + "_" + obj + "_relativeangles_tester.p" #this works but only for 90...

ylim_max = 0.5 if obj == "Toaster" else 2.0



aosm_data = pickle.load(open(data_dir+"/"+aosm_file,"rb"))

X_success = aosm_data["success"]
X_failure= aosm_data["failure"]

x, y, a_list = zip(*X_success)
x = np.array(x)
y = np.array(y)

print("original a:",a_list)
a = list(map(lambda x: x%360, a_list))
a = np.array(a)
print("remapped a:",a)
print("angles min and max:",a.min(),a.max())
#print(a.min())
#print(a.max())
test_angles = np.linspace(a.min(),a.max(),num=10)

########## SHOW DATA ###############
#to place vectors at each point in accordance with test angle
example_vectors = []
for collected_angle in a_list:
  example_angle = collected_angle
  rotated_vector = rotate([0,0], [1,0], -1*example_angle) 
  example_vectors.append(rotated_vector)
example_vectors = np.array(example_vectors)
ev_x = example_vectors[:,0]
ev_y = example_vectors[:,1]

quiver_scale = 4.5
obj_rot = 90
obj_vector = rotate([0,0], [1,0], -1*obj_rot) #multiply by negative one since we need clockwise

plt.xlim([-2,2])
plt.ylim([-2,ylim_max])
plt.scatter(0, 0, s=10, facecolor='green')
plt.scatter(x,y, s=2, facecolor='red')
plt.quiver(x,y, ev_x, ev_y, angles='xy', scale_units='xy', scale=quiver_scale)
plt.quiver(0, 0, obj_vector[0], obj_vector[1], angles='xy', scale_units='xy', scale=quiver_scale)
plt.show()

####### SHOW RANDOM SAMPLES #######
num_samples = 100

test_angle = 10000 #not relevant for this part, random value
bandwidth = 0.13
_, _, _, kde_skl = kde2D(x, y, a, test_angle, bandwidth)
sampled_poses = kde_skl.sample((num_samples,1))
sampled_poses = np.squeeze(sampled_poses,axis=1)
print("Sampled poses:", sampled_poses.shape)
example_vectors = []
for sampled_pose in sampled_poses:
  print("cur sampled pose:", sampled_pose)
  example_angle = sampled_pose[2]
  rotated_vector = rotate([0,0], [1,0], -1*example_angle) 
  example_vectors.append(rotated_vector)
example_vectors = np.array(example_vectors)
ev_x = example_vectors[:,0]
ev_y = example_vectors[:,1]

sample_x = sampled_poses[:,0]
sample_y = sampled_poses[:,1]

quiver_scale = 4.5
obj_rot = 90
obj_vector = rotate([0,0], [1,0], -1*obj_rot) #multiply by negative one since we need clockwise

plt.xlim([-2,2])
plt.ylim([-2,ylim_max])
plt.scatter(0, 0, s=10, facecolor='green')
plt.scatter(sample_x,sample_y, s=2, facecolor='red')
plt.quiver(sample_x,sample_y, ev_x, ev_y, angles='xy', scale_units='xy', scale=quiver_scale)
plt.quiver(0, 0, obj_vector[0], obj_vector[1], angles='xy', scale_units='xy', scale=quiver_scale)
plt.show()

for test_angle in test_angles:
  print("test angle:",test_angle)

  global_robot_rot = test_angle #use this for plotting vector
  print("global angle:",global_robot_rot)

  rotated_vector = rotate([0,0], [1,0], -1*global_robot_rot) #multiply by negative one since we need clockwise
  print("rotated vector:",rotated_vector)

  xx, yy, zz, kde_skl = kde2D(x, y, a, test_angle, bandwidth)

  plt.xlim([-2,2])
  plt.ylim([-2,ylim_max])
  plt.pcolormesh(xx, yy, zz)

  plt.scatter(0, 0, s=10, facecolor='green')

  rotated_vector_list = np.array([rotated_vector]*x.shape[0])

  rvl_x = rotated_vector_list[:,0]
  rvl_y = rotated_vector_list[:,1]
  if viz_points:
    plt.scatter(x, y, s=2, facecolor='red')
    plt.quiver(x, y, rvl_x, rvl_y, angles='xy', scale_units='xy', scale=quiver_scale)
    plt.quiver(0, 0, obj_vector[0], obj_vector[1], angles='xy', scale_units='xy', scale=quiver_scale)
  plt.show()