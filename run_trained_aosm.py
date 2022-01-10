from ai2thor.controller import Controller
import pickle
from sklearn.neighbors import KernelDensity
import numpy as np
import random

def kde2D(x, y, a, bandwidth): 
    """Build 2D kernel density estimate (KDE)."""

    xy_train  = np.vstack([x, y, a]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, kernel = "gaussian")
    kde_skl.fit(xy_train)

    return(kde_skl)

def run_train_aosm(controller,action_object,gridSize=0.25,floorPlan='FloorPlan1', data_dir = "./aosm_data/"):
	controller.reset('FloorPlan1')
	controller.step(
		action="InitialRandomSpawn",
		randomSeed=1000,
		forceVisible=True,
		numPlacementAttempts=50,
		placeStationary=True
	)

	action, obj = action_object

	#Load learned KDE data
	aosm_file = action + "_" + obj + "_relativeangles_tester.p"
	#aosm_file = action + "_" + obj + "_relativeangles_tester.p" #this works but only for 90...
	aosm_data = pickle.load(open(data_dir+"/"+aosm_file,"rb"))

	X_success = aosm_data["success"]
	X_failure= aosm_data["failure"]

	x, y, a_list = zip(*X_success)
	x = np.array(x)
	y = np.array(y)

	a = list(map(lambda x: x%360, a_list))
	a = np.array(a)

	################## Find object reference
	event = controller.step(action='GetReachablePositions')
	all_objs = event.metadata['objects']
	random.shuffle(all_objs)
	for possible_obj in all_objs:
		#get an object of right category but not assigned yet
		if obj in possible_obj['objectId']:
			obj_ref = possible_obj
			break
	#print("NEW OBJECT BINDING!",obj_ref['objectId'])

	#object loc
	obj_loc = obj_ref['position']
	obj_rot = obj_ref['rotation']
	print("obj loc:", obj_loc)
	print("obj rot:",obj_rot)


	################## fit KDE
	bandwidth = 0.13
	kde_skl = kde2D(x, y, a, 0.13)

	################## Sample points from KDE
	num_attempts = 100
	sampled_poses = kde_skl.sample((num_attempts,1))
	sampled_poses = np.squeeze(sampled_poses,axis=1)
	print("Sampled poses!:")
	print(sampled_poses)

	#################3Try each sample and see if it works
	for sample_pose in sampled_poses:
		#transform sample pose
		sample_pos = [sample_pose[0],sample_pose[1]]
		sample_rot = sample_pose[2]
		print("try this sample pos and rot",sample_pos, sample_rot)
		global_sample_pos_x = obj_loc['x'] - sample_pose[1]
		global_sample_pos_z = obj_loc['z'] + sample_pose[0]
		global_sample_rot = sample_rot
		print("try this global pos and rot",global_sample_pos_x, global_sample_pos_z, global_sample_rot)

		try:
			y_pos = event.metadata['agent']['position']['y']
			horizon = 0
			event = controller.step(action='Teleport', x=global_sample_pos_x, y=y_pos, z=global_sample_pos_z, rotation=dict(x=0.0, y=global_sample_rot, z=0.0), horizon=horizon,raise_for_failure=True)
			event = controller.step(action='Teleport', x=global_sample_pos_x, y=y_pos, z=global_sample_pos_z, rotation=dict(x=0.0, y=global_sample_rot, z=0.0), horizon=horizon,raise_for_failure=True)
			print("succeed in going!")
			input("blah")
		except:
			print("MASSIVE FAILURE22222")
			continue

if __name__ == "__main__":
	data_dir = "./aosm_data"
	#action_object = ["PickupObject", "Knife"]
	#action_object = ["ToggleObjectOn", "Toaster"]
	#action_object = ["PickupObject", "Potato"]

	action_object = ["PickupObject", "Mug"]
	#action_object = ["ToggleObjectOn", "CoffeeMachine"]



	floor_idx = 1
	floor_scene = "FloorPlan"+str(floor_idx)

	gridSize = 0.25

	img_width = 500
	img_height = 500

	controller = Controller(scene=floor_scene, gridSize=gridSize,width=500,height=500,renderObjectImage=True,renderClassImage=True,renderDepthImage=True,snapToGrid=False)
	run_train_aosm(controller,action_object,gridSize=0.25,floorPlan=floor_scene, data_dir=data_dir)

	#rotate_test()