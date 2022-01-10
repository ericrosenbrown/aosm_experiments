#from viz_room import viz_room
#from viz_room import viz_object_action_poses
from ai2thor.controller import Controller
import pickle
from collections import defaultdict
import json 
import numpy as np
import math
import copy
import random
import math
from tqdm import tqdm
import os
import pickle

from PIL import Image

def nearest_neighbor(p,poses):
	#given a 2d point p ([x,y]) and a planning tree (list consisting of Nodes), return the node that is closest to p
	best_point = None #default point is root
	best_distance = float("inf")
	for point in poses:
		dist = np.linalg.norm(np.array([point['x'],point['z']])-np.array([p['x'],p['z']]))
		if dist < best_distance:
			best_point = point
			best_distance= dist
	return(best_point)

def action_spots(p,poses,thresh=1.5): #1.25 is a close threshold
	#given a 2d point p ([x,y]) and a planning tree (list consisting of Nodes), return the node that is closest to p
	good_enough_spots = []
	for point in poses:
		dist = np.linalg.norm(np.array([point['x'],point['z']])-np.array([p['x'],p['z']]))
		if dist < thresh:
			good_enough_spots.append([point['x'],point['z']])
	return(good_enough_spots)

def action_spots_circle(p,thresh=1.5): #1.25 is a close threshold
	perc = random.random()
	theta = perc * math.pi * 2
	new_x = p['x'] + thresh * math.sin(theta)
	new_z = p['z'] + thresh * math.cos(theta)
	return[[new_x,new_z]]

def action_spots_noisy(p,poses,thresh=1.5,noisy=0.125):
	#given a 2d point p ([x,y]) and a planning tree (list consisting of Nodes), return the node that is closest to p
	good_enough_spots = []
	for point in poses:
		dist = np.linalg.norm(np.array([point['x'],point['z']])-np.array([p['x'],p['z']]))
		if dist < thresh:
			good_enough_spots.append([point['x'] + random.uniform(-1*noisy, noisy),point['z']+ random.uniform(-1*noisy, noisy)])
	return(good_enough_spots)		


def get_angle(robot_pos,object_pos):
	#get angle for robot_pos to face object_pos
	y = (object_pos["x"] - robot_pos["x"])
	x = object_pos["z"] - robot_pos["z"]

	degree = math.atan2(y,x)*180/(math.pi)
	if degree < 0:
		degree = 360 + degree
	return(degree)

def collect_naive_data(manip_plan,num_attempts=20,gridSize=0.25):
	naive_stats = {}
	for i in tqdm(range(1,31)):
		floorKey = i
		naive_stats[i] = []
		for attempt in range(num_attempts):
			try:
				controller = Controller(scene="FloorPlan"+str(i), gridSize=gridSize,width=500,height=500,renderObjectImage=True,renderClassImage=True,renderDepthImage=True)
				total_steps = plan_navigation(controller,manip_plan,gridSize,floorPlan="FloorPlan"+str(i),seed=attempt)
				naive_stats[i].append(total_steps)
			except:
				print("I riased an error, oops",i)
				controller.stop()
	#print(naive_stats)
	pickle.dump(naive_stats,open("naive_stats.p","wb"))

def object_relative_pose(obj_loc, obj_rot, robot_loc, robot_rot, horizon):
	rel_pos_x = obj_loc['x'] - robot_loc['x']
	rel_pos_z = obj_loc['z'] - robot_loc['z']

	rel_rot_y = obj_rot['y'] - robot_rot['y']


	#return([rel_pos_x, rel_pos_z, rel_rot_y, horizon])
	return([rel_pos_x, rel_pos_z], rel_rot_y)

def object_relative_pose2(obj_loc, obj_rot, robot_loc, robot_rot, horizon):
	rel_pos_x = robot_loc['x'] - obj_loc['x']
	rel_pos_z = robot_loc['z'] - obj_loc['z']

	rel_rot_y = robot_rot['y'] - obj_rot['y']


	#return([rel_pos_x, rel_pos_z, rel_rot_y, horizon])
	return([rel_pos_x, rel_pos_z], rel_rot_y)

def object_relative_pose3(obj_loc, obj_rot, robot_loc, robot_rot, horizon):
	rel_pos_x = robot_loc['x'] - obj_loc['x']
	rel_pos_z = robot_loc['z'] - obj_loc['z']

	rel_rot_y = robot_rot['y'] + obj_rot['y']


	#return([rel_pos_x, rel_pos_z, rel_rot_y, horizon])
	return([rel_pos_x, rel_pos_z], rel_rot_y)


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


def collect_action_object_data(amount,controller,action_object,gridSize=0.25,floorPlan='FloorPlan1', data_dir = "./complete_data/", sample_method="heuristic"):
	action, obj = action_object

	#agent_pos (relative to object), agent_rot (relative to object)
	aosm_dict = {
	"success": [],
	"failure": []}

	for i in range(amount):
		print("success:", len(aosm_dict["success"]), "failure", len(aosm_dict["failure"]))

		controller.reset('FloorPlan1')
		controller.step(
			action="InitialRandomSpawn",
			randomSeed=i,
			forceVisible=True,
			numPlacementAttempts=50,
			placeStationary=True
		)

		event = controller.step(action='GetReachablePositions')
		room = event.metadata['actionReturn']

		#assign an object to be the referent
		#Find object reference
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


		if sample_method == "heuristic":
			#nearest free pose to object
			#close_enough_free_spots = action_spots(obj_loc,room)
			close_enough_free_spots = action_spots_noisy(obj_loc,room)
			#close_enough_free_spots = action_spots_circle(obj_loc)
			#print("CLOSE ENOUGH FREE SPOTS:",close_enough_free_spots)
			robot_goal_xz = random.choice(close_enough_free_spots)


			horizons  = [0]#[-30,0,30,60]
			horizon = random.choice(horizons)
			robot_goal_xz = random.choice(close_enough_free_spots)
			y_pos = event.metadata['agent']['position']['y']

			##### TELEPORT MOVE #######
			start_degree = 90
			try:
				event = controller.step(action='Teleport', x=robot_goal_xz[0], y=y_pos, z=robot_goal_xz[1], rotation=dict(x=0.0, y=start_degree, z=0.0), horizon=horizon,raise_for_failure=True)
				event = controller.step(action='Teleport', x=robot_goal_xz[0], y=y_pos, z=robot_goal_xz[1], rotation=dict(x=0.0, y=start_degree, z=0.0), horizon=horizon,raise_for_failure=True)
			except:
				print("MASSIVE FAILURE111111")
				aosm_dict['failure'].append([robot_goal_xz[0],robot_goal_xz[1],start_degree])
				continue

			#input("just moved to clsoe location, not currently facing object")

			agent = event.metadata['agent']
			global_robot_loc = agent['position']
			global_robot_rot = agent['rotation']

			y_degree = get_angle({"x":global_robot_loc['x'],"z":global_robot_loc['z']},obj_loc)

			#y_degree = random.choice([225,270,315])
			#print("my y degree" + str(y_degree))
			#input("wait")
			try:
				event = controller.step(action='Teleport', x=robot_goal_xz[0], y=y_pos, z=robot_goal_xz[1], rotation=dict(x=0.0, y=y_degree, z=0.0), horizon=horizon,raise_for_failure=True)
				event = controller.step(action='Teleport', x=robot_goal_xz[0], y=y_pos, z=robot_goal_xz[1], rotation=dict(x=0.0, y=y_degree, z=0.0), horizon=horizon,raise_for_failure=True)
			except:
				aosm_dict['failure'].append([robot_goal_xz[0],robot_goal_xz[1],y_degree])
				print("MASSIVE FAILURE22222")
				continue
		elif sample_method == "random":
			event = controller.step(action="GetReachablePositions")
			room = event.metadata['actionReturn']

			y_pos = event.metadata['agent']['position']['y']
			horizon = 0 

			orientations = [0,90,180,270]
			random.shuffle(room)
			xyz = room[0]
			x = xyz['x']
			z = xyz['z']

			random.shuffle(orientations)
			y_orientation = orientations[0]

			try:
				event = controller.step(action='Teleport', x=x, y=y_pos, z=z, rotation=dict(x=0.0, y=y_orientation, z=0.0), horizon=horizon,raise_for_failure=True)
				event = controller.step(action='Teleport', x=x, y=y_pos, z=z, rotation=dict(x=0.0, y=y_orientation, z=0.0), horizon=horizon,raise_for_failure=True)
			except:
				aosm_dict['failure'].append([x,z,y_orientation])
				print("MASSIVE FAILURE22222")
				continue




		#input("now im facing the object?")
		agent = event.metadata['agent']
		global_robot_loc = agent['position']
		global_robot_rot = agent['rotation']

		#pose3 should be right one
		relative_robot_pose, rel_rot_y = object_relative_pose3(obj_loc, obj_rot, global_robot_loc, global_robot_rot, horizon)
		#print("obj loc:" + str(obj_loc))
		print("obj rot:" + str(obj_rot))
		#print("robot loc:" + str(global_robot_loc))
		print("robot rot:" + str(global_robot_rot))
		print(rel_rot_y)
		hardcoded_rot = 90 #hard coded angle for object-oriented frame downward
		rel_rot_y_corrected = rel_rot_y - hardcoded_rot #this is to get it into object centric frame facing down
		print("relative pose:", str(relative_robot_pose))
		rotated_relative_pose = rotate([0,0], relative_robot_pose, -1*obj_rot['y'])
		print("rotated relative pose:", rotated_relative_pose)
		rerotated_relative_pose = rotate([0,0],rotated_relative_pose,obj_rot['y'])
		#print("rerotated relative pose:", rerotated_relative_pose)
		rotated_relative_pose = list(rotated_relative_pose)
		#input("")
		if action == "PickupObject":
			try:
				event = controller.step(action='PickupObject',
					objectId=obj_ref['objectId'],
					raise_for_failure=True)
				print("Picked up:",obj_ref['objectId'])
				aosm_dict['success'].append(rotated_relative_pose + [rel_rot_y_corrected])
			except:
				print("failed to pick up")
				aosm_dict['failure'].append(rotated_relative_pose + [rel_rot_y_corrected])

		elif action == "SliceObject":
			event = controller.step(action='SliceObject',
				objectId=obj_ref['objectId'],
				raise_for_failure=True)
			print("Sliced:",obj_ref['objectId'])

		elif action == "PutObject":
			print("Put:",heldObject)
			print("into:",obj_ref['objectId'])
			event = controller.step("LookDown")
			event = controller.step(action='PutObject',
				receptacleObjectId=obj_ref['objectId'],
				objectId=heldObject,
				raise_for_failure=True)
			print("Put down:",obj_ref['objectId'])

		elif action == "ToggleObjectOn":
			try:
				event = controller.step(action='ToggleObjectOn',
					objectId=obj_ref['objectId'],
					raise_for_failure=True)
				print("Toggle on:",obj_ref['objectId'])
				aosm_dict['success'].append(rotated_relative_pose + [rel_rot_y_corrected])
			except:
				print("failed to toggle on")
				aosm_dict['failure'].append(rotated_relative_pose + [rel_rot_y_corrected])

		elif action == "ToggleObjectOff":
			event = controller.step(action='ToggleObjectOff',
				objectId=obj_ref['objectId'],
				raise_for_failure=True)
			print("Toggle off:",obj_ref['objectId'])

		print("================")
		pickle.dump(aosm_dict, open(data_dir + "/" + action+"_" + obj+ "_floor1_"+sample_method+"_"+str(amount)+".p", "wb" ) )


	controller.stop()


if __name__ == "__main__":
	data_dir = "./complete_data"
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

	sample_method = "heuristic" #"random"

	amount = 25
	controller = Controller(scene=floor_scene, gridSize=gridSize,width=500,height=500,renderObjectImage=False,renderClassImage=False,renderDepthImage=False,snapToGrid=False)
	collect_action_object_data(amount,controller,action_object,gridSize=0.25,floorPlan=floor_scene, data_dir=data_dir, sample_method=sample_method)

	#rotate_test()