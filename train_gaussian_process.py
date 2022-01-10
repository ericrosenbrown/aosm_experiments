import pickle
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

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
	gpr = GaussianProcessClassifier().fit(X,Y)
	return(gpr)

def visualize_gp(gpr):
	#range of rotations [0,45,90,135,180,225,270,315]
	#range of horizons [-30,0,30,60]
	#range of xz pos: [-3, 3]
	pos_min = -1
	pos_max = 2
	num = 20

	rotations = [225,270,315]#[0,45,90,135,180,225,270,315]
	horizons = [0]#[-30,0,30,60]
	for rot in rotations:
		preds = []
		xx = np.linspace(-2,2,num)
		zz = np.linspace(0,1,num)
		for x in xx:
			for z in zz:
				new_pred = gpr.predict([[x,z,rot]])
				print(new_pred)
				preds.append(new_pred)
				#print(preds[-1])
		preds = np.array(preds)
		preds = np.reshape(preds, [num,num])
		h = plt.contourf(xx, zz, preds)
		cbar = plt.colorbar(h)
		plt.axis('scaled')
		print("rotation:" + str(rot))
		plt.show()



if __name__ == "__main__":
	data_dir = "./aosm_data"
	#action, obj = ["PickupObject","Knife"]
	action, obj = ["ToggleObjectOn", "Toaster"]
	aosm_file = action + "_" + obj + ".p"

	aosm_data = pickle.load(open(data_dir+"/"+aosm_file,"rb"))
	gpr = train_gp(aosm_data)
	visualize_gp(gpr)
