import pickle
import matplotlib.pyplot as plt

def viz_aosm(aosm_data, degree):
	X_success = aosm_data["success"]
	X_failure= aosm_data["failure"]
	print(X_success)

	for x in X_success:
		#if x[-1] == degree:
		plt.scatter(x[0], x[1], color='green')
	for x in X_failure:
		#if x[-1] == degree:
		plt.scatter(x[0], x[1], color='red')
	plt.show()


if __name__ == "__main__":
	data_dir = "./aosm_data"
	#action, obj = ["PickupObject","Knife"]
	action, obj = ["ToggleObjectOn", "Toaster"]
	aosm_file = action + "_" + obj + "_directlook.p"

	aosm_data = pickle.load(open(data_dir+"/"+aosm_file,"rb"))
	for degree in [225,270,315]:
		gpr = viz_aosm(aosm_data, degree)
