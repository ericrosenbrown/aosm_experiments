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

from path_planner import rrt_path_planner, rrt_path_planner_multigoal

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

def action_spots(p,poses,thresh=1.25):
	#given a 2d point p ([x,y]) and a planning tree (list consisting of Nodes), return the node that is closest to p
	good_enough_spots = []
	for point in poses:
		dist = np.linalg.norm(np.array([point['x'],point['z']])-np.array([p['x'],p['z']]))
		if dist < thresh:
			good_enough_spots.append([point['x'],point['z']])
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
	print(naive_stats)
	pickle.dump(naive_stats,open("naive_stats.p","wb"))




#def plan_navigation(manip_plan,gridSize=0.25,floorPlan='FloorPlan2',seed=0):
def plan_navigation(controller,manip_plan,gridSize=0.25,floorPlan='FloorPlan2',seed=0):
	fname = "baseline_images/"+floorPlan+"_toast_1/"
	counter = 0
	save_images = False

	event = controller.step(action="InitialRandomSpawn",
	randomSeed=seed,
	forceVisible=True,
	numPlacementAttempts=5,
	placeStationary=True)

	event = controller.step(action='GetReachablePositions')
	room = event.metadata['actionReturn']

	#input("wait")

	heldObject = None

	important_obj_refs = {}

	controller.step("LookDown")
	controller.step("LookDown")
	if save_images:
		im = Image.fromarray(event.frame)
		im.save(fname+"rgb/"+str(counter)+".png")

		mask_im = Image.fromarray(event.instance_segmentation_frame)
		mask_im.save(fname+"mask/"+str(counter)+".png")

	counter += 1

	for manip_step in manip_plan:
		action, obj, obj_highlevel_ref = manip_step

		obj_ref = None
		#referenced object has been assigned previously
		if obj_highlevel_ref in important_obj_refs.keys():
			obj_ref_id = important_obj_refs[obj_highlevel_ref]

			all_objs = event.metadata['objects']
			for possible_obj in all_objs:
				if possible_obj['objectId'] == obj_ref_id:
					obj_ref = possible_obj
			print("i've seen this object before:",obj_ref_id)


		#assign an object to be the referent, different than one in the existing setup
		else:
			#Find object reference
			all_objs = event.metadata['objects']
			random.shuffle(all_objs)
			for possible_obj in all_objs:
				#get an object of right category but not assigned yet
				if obj in possible_obj['objectId'] and possible_obj['objectId'] not in important_obj_refs.values():
					important_obj_refs[obj_highlevel_ref] = possible_obj['objectId']
					obj_ref = possible_obj
					break
			print("NEW OBJECT BINDING!",possible_obj['objectId'])

		#object loc
		object_loc = obj_ref['position']

		#nearest free pose to object
		#closest_free_spot = nearest_neighbor(object_loc,room)
		close_enough_free_spots = action_spots(object_loc,room)
		print("CLOS ENOUGH FREE SPOTS:",close_enough_free_spots)


		#########event = controller.step(action='TeleportFull', x=closest_free_spot['x'], y=closest_free_spot['y'], z=closest_free_spot['z'], rotation=dict(x=0.0, y=degree, z=0.0), horizon=30.0,raise_for_failure=True)
		clean_room = [[pos_dict['x'],pos_dict['z']] for pos_dict in room]
		agent = event.metadata['agent']
		robot_pose = [agent['position']['x'],agent['position']['y'],agent['position']['z'],0]
		plan = rrt_path_planner_multigoal(clean_room,robot_pose,close_enough_free_spots,threshold=0.25)
		plan.reverse()

		print("plan:",plan)
		degree = get_angle({"x":plan[-1][0],"z":plan[-1][1]},object_loc)
		goal_pose = [plan[-1][0],room[0]["y"],plan[-1][1],degree]

		##### TELEPORT MOVE #######
		'''	
		for pos in plan:
			#print("go to:",pos)
			#print(pos[0],goal_pose[1],pos[1],0.0,goal_pose[3],0.0,30.0)
			#input("")
			try:
				event = controller.step(action='TeleportFull', x=pos[0], y=goal_pose[1], z=pos[1], rotation=dict(x=0.0, y=goal_pose[3], z=0.0), horizon=30.0,raise_for_failure=True, standing=True)
				if save_images:
					im = Image.fromarray(event.frame)
					im.save(fname+"rgb/"+str(counter)+".png")

					mask_im = Image.fromarray(event.instance_segmentation_frame)
					mask_im.save(fname+"mask/"+str(counter)+".png")
				counter += 1
			except:
				pass
		print("finished going!")
		input("try manip")
		'''
		###### ACTUAYL MOVE ####
		for i in range(1,len(plan)):
			print("go to:",plan[i-1],plan[i])
			robot_from = {"x":plan[i-1][0], "z":plan[i-1][1]}
			robot_to = {"x":plan[i][0], "z":plan[i][1]}

			robot_dist = np.linalg.norm(np.array(plan[i-1])-np.array(plan[i]))
			robot_degree_need = get_angle(robot_from,robot_to)

			agent = event.metadata['agent']
			print("robot position:",agent["position"])
			robot_degree_cur = agent['rotation']['y']
			rotation_delta = robot_degree_need - robot_degree_cur
			#keep angles between 0-360, increments of 45 for discrete action space
			print("robot_degree_cur:",robot_degree_cur)
			print("Robot need to be:",robot_degree_need)
			print("rotate:",rotation_delta)
			corrected_rotation_delta = rotation_delta + 360 if rotation_delta < 0 else rotation_delta
			corrected_rotation_delta = corrected_rotation_delta - (corrected_rotation_delta%45)
			print("corrected_rotation_delta:",corrected_rotation_delta)
			print("robot move:",robot_dist)
			input("")
			event = controller.step(
				action="RotateRight",
				degrees=corrected_rotation_delta)
			input("move")
			event = controller.step(
				action="MoveAhead",
				moveMagnitude=robot_dist)

		print("finsihed going!")
		agent = event.metadata['agent']
		agent_pos = agent["position"]
		robot_degree_need = get_angle(agent_pos,object_loc)
		robot_degree_cur = agent['rotation']['y']
		rotation_delta = robot_degree_need - robot_degree_cur
		input("findal correct")
		event = controller.step(
				action="RotateRight",
				degrees=rotation_delta)

		print("robot position:",agent["position"])

		#################

		#input("before manip")
		if action == "PickupObject":
			event = controller.step(action='PickupObject',
				objectId=obj_ref['objectId'],
				raise_for_failure=True)
			print("Picked up:",obj_ref['objectId'])
			heldObject = obj_ref['objectId']

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
			event = controller.step(action='ToggleObjectOn',
				objectId=obj_ref['objectId'],
				raise_for_failure=True)
			print("Toggle on:",obj_ref['objectId'])

		elif action == "ToggleObjectOff":
			event = controller.step(action='ToggleObjectOff',
				objectId=obj_ref['objectId'],
				raise_for_failure=True)
			print("Toggle off:",obj_ref['objectId'])

		if save_images:
			im = Image.fromarray(event.frame)
			im.save(fname+"rgb/"+str(counter)+".png")


			mask_im = Image.fromarray(event.instance_segmentation_frame)
			mask_im.save(fname+"mask/"+str(counter)+".png")
		counter += 1


		#input("wait")
	controller.stop()
	return(counter)


if __name__ == "__main__":

	toast_plan = [["PickupObject","Knife","Knife0"],
	["SliceObject","Bread","Bread0"],
	["PutObject","CounterTop","CounterTop0"],
	["PickupObject","BreadSliced","BreadSliced0"],
	["PutObject","Toaster","Toaster0"],
	["ToggleObjectOn","Toaster","Toaster0"],
	["ToggleObjectOff","Toaster","Toaster0"],
	["PickupObject","BreadSliced","BreadSliced0"],
	]

	egg_plan = [["PickupObject","Pan","Pan0"],
	["PutObject","StoveBurner","Stove0"],
	["PickupObject","Egg","Egg0"],
	["PutObject","Pan","Pan0"],
	["SliceObject","Egg","Egg0"],
	["ToggleObjectOn","StoveKnob","StoveKnob0"],
	["ToggleObjectOn","StoveKnob","StoveKnob1"],
	["ToggleObjectOn","StoveKnob","StoveKnob2"],
	["ToggleObjectOn","StoveKnob","StoveKnob3"],
	["ToggleObjectOn","StoveKnob","StoveKnob4"],
	["ToggleObjectOn","StoveKnob","StoveKnob5"],
	["ToggleObjectOff","StoveKnob","StoveKnob0"],
	["ToggleObjectOff","StoveKnob","StoveKnob1"],
	["ToggleObjectOff","StoveKnob","StoveKnob2"],
	["ToggleObjectOff","StoveKnob","StoveKnob3"],
	["ToggleObjectOff","StoveKnob","StoveKnob4"],
	["ToggleObjectOff","StoveKnob","StoveKnob5"],
	]

	floor_plans = [1,2,3,4,5,6,7,8,9,10,11]
	i = 1
	gridSize = 0.25

	controller = Controller(scene="FloorPlan"+str(i), gridSize=gridSize,width=500,height=500,renderObjectImage=True,renderClassImage=True,renderDepthImage=True)
	plan_navigation(controller,toast_plan,gridSize=0.25,floorPlan="FloorPlan"+str(i),seed=3420)
	#collect_naive_data(toast_plan,gridSize=0.25)

	#rotate_test()
