import pickle
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def train_gp(aosm_data):
	X_success = aosm_data["success"]
	X_failure= aosm_data["failure"]
	print(X_success)

	Y_success = [1]*len(X_success)
	Y_failure = [0]*len(X_failure)

	X = X_success + X_failure
	Y = Y_success + Y_failure


	#kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3))
	#gpr = GaussianProcessRegressor(kernel=kernel, alpha=0).fit(X,Y)
	print("fittin!")
	print(np.array(X_success).shape)
	kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_success)
	#print("hey" + str(kde.score_samples(X)))
	return(kde)

def visualize_gp(gpr, aosm_data):
	#range of rotations [0,45,90,135,180,225,270,315]
	#range of horizons [-30,0,30,60]
	#range of xz pos: [-3, 3]
	pos_min = -1
	pos_max = 2
	num = 100

	rotations = [225,270,315]#[0,45,90,135,180,225,270,315]
	horizons = [0]#[-30,0,30,60]
	for rot in rotations:
		preds = []
		xx = np.linspace(-1.5,1.5,num)
		zz = np.linspace(0.4,1,num)
		for x in xx:
			for z in zz:
				new_pred = gpr.score_samples([[x,z,rot]])
				#print(new_pred)
				preds.append(new_pred)
				#print(preds[-1])
		preds = np.array(preds)
		preds = np.reshape(preds, [num,num])
		h = plt.contourf(xx, zz, preds)
		cbar = plt.colorbar(h)
		print("rotation:" + str(rot))

		viz_aosm(aosm_data, rot)

		plt.show()

def viz_aosm(aosm_data, degree):
	X_success = aosm_data["success"]
	X_failure= aosm_data["failure"]
	print(X_success)

	for x in X_success:
		if x[-1] == degree:
			plt.scatter(x[0], x[1], color='green')
	for x in X_failure:
		if x[-1] == degree:
			plt.scatter(x[0], x[1], color='red')
	#plt.show()



if __name__ == "__main__":
	data_dir = "./aosm_data"
	#action, obj = ["PickupObject","Knife"]
	action, obj = ["ToggleObjectOn", "Toaster"]
	aosm_file = action + "_" + obj + ".p"

	aosm_data = pickle.load(open(data_dir+"/"+aosm_file,"rb"))
	gpr = train_gp(aosm_data)
	#for degree in [225,270,315]:

	visualize_gp(gpr, aosm_data)
