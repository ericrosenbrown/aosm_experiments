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

def run_train_aosm(amount,controller,action_object,gridSize=0.25,floorPlan='FloorPlan1', data_dir = "./aosm_data/"):
	action, obj = action_object

	#Load learned KDE data
	aosm_file = action + "_" + obj + "_floor1_heuristic_"+str(amount)+".p"
	#aosm_file = action + "_" + obj + "_relativeangles_tester.p" #this works but only for 90...
	aosm_data = pickle.load(open(data_dir+"/"+aosm_file,"rb"))

	X_success = aosm_data["success"]
	X_failure= aosm_data["failure"]

	x, y, a_list = zip(*X_success)
	x = np.array(x)
	y = np.array(y)

	a = list(map(lambda x: x%360, a_list))
	a = np.array(a)

	################## fit KDE
	bandwidth = 0.13 #0.13
	kde_skl = kde2D(x, y, a, bandwidth)

	################## Sample points from KDE
	num_poses_to_possibly_try= 1000
	sampled_poses = kde_skl.sample((num_poses_to_possibly_try,1))
	sampled_poses = np.squeeze(sampled_poses,axis=1)

	max_num_attempts = 50
	num_success = 0
	num_attempts = 0

	#################3Try each sample and see if it works
	for idx, sample_pose in enumerate(sampled_poses):
		controller.reset('FloorPlan1')
		controller.step(
			action="InitialRandomSpawn",
			randomSeed = idx,
			forceVisible=True,
			numPlacementAttempts=50,
			placeStationary=True
		)

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



		print("num success:",num_success)
		#transform sample pose
		sample_pos = [sample_pose[0],sample_pose[1]]
		sample_rot = sample_pose[2]

		global_sample_pos_x = obj_loc['x'] - sample_pose[1]
		global_sample_pos_z = obj_loc['z'] + sample_pose[0]
		global_sample_rot = sample_rot


		try:
			y_pos = event.metadata['agent']['position']['y']
			horizon = 0
			event = controller.step(action='Teleport', x=global_sample_pos_x, y=y_pos, z=global_sample_pos_z, rotation=dict(x=0.0, y=global_sample_rot, z=0.0), horizon=horizon,raise_for_failure=True)
			event = controller.step(action='Teleport', x=global_sample_pos_x, y=y_pos, z=global_sample_pos_z, rotation=dict(x=0.0, y=global_sample_rot, z=0.0), horizon=horizon,raise_for_failure=True)
			print("succeed in going!")
		except:
			print("MASSIVE FAILURE22222")
			continue

		if action == "ToggleObjectOn":
			num_attempts +=1
			try:
				event = controller.step(action='ToggleObjectOn',
					objectId=obj_ref['objectId'],
					raise_for_failure=True)
				print("Toggle on:",obj_ref['objectId'])
				num_success += 1
			except:
				print("failed to toggle on")
		elif action == "PickupObject":
			num_attempts +=1
			try:
				event = controller.step(action='PickupObject',
					objectId=obj_ref['objectId'],
					raise_for_failure=True)
				print("Picked up:",obj_ref['objectId'])
				num_success += 1
			except:
				print("failed to pick up")

		if num_attempts == max_num_attempts:
			break



	print("amount classifier had:",amount)
	print("Num success:",num_success)
	print("Total attempts:",num_attempts)
	print("success rate:",float(num_success)/num_attempts)

if __name__ == "__main__":
	data_dir = "./complete_data"
	#action_object = ["PickupObject", "Knife"]
	#action_object = ["ToggleObjectOn", "Toaster"]
	#action_object = ["PickupObject", "Potato"]

	action_object = ["PickupObject", "Mug"]
	#action_object = ["ToggleObjectOn", "CoffeeMachine"]

	#amounts: 25,100
	amount = 25


	floor_idx = 1
	floor_scene = "FloorPlan"+str(floor_idx)

	gridSize = 0.25

	img_width = 500
	img_height = 500

	controller = Controller(scene=floor_scene, gridSize=gridSize,width=500,height=500,renderObjectImage=True,renderClassImage=True,renderDepthImage=True,snapToGrid=False)
	run_train_aosm(amount,controller,action_object,gridSize=0.25,floorPlan=floor_scene, data_dir=data_dir)

	#rotate_test()