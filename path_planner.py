import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

def nearest_neighbor(p,ptree):
    #given a 2d point p ([x,y]) and a planning tree (list consisting of Nodes), return the node that is closest to p
    best_point = None #default point is root
    best_distance = float("inf")
    for point in ptree:
        dist = np.linalg.norm(np.array(point.p)-np.array(p))
        if dist < best_distance:
            best_point = point
            best_distance= dist
    return(best_point)

def nearest_neighbor_free(p,room):
    #given a 2d point p ([x,y]) and a planning tree (list consisting of Nodes), return the node that is closest to p
    best_point = None #default point is root
    best_distance = float("inf")
    for point in room:
        dist = np.linalg.norm(np.array(point)-np.array(p))
        if dist < best_distance:
            best_point = point
            best_distance= dist
    return(best_point,best_distance)

def dd():
	return defaultdict(list)

def steer(new,nearest,room,grid_size=0.25):
	#return a point extending from nearest to new that is at most length threshold

	if new[0] >= nearest.p[0] + grid_size/2:
		#print("GO RIGHT")
		xd = grid_size
	elif new[0] < nearest.p[0] - grid_size/2:
		#print("GO LEFT")
		xd = -1*grid_size
	else:
		xd = 0
	if new[1] >= nearest.p[1] + grid_size/2:
		#print("GO UP")
		zd = grid_size
	elif new[1] < nearest.p[1] - grid_size/2:
		#print("GO DOWN")
		zd = -1*grid_size
	else:
		zd = 0

	diff_p = [nearest.p[0]+xd,nearest.p[1]+zd]
	return(nearest_neighbor_free(diff_p,room)[0])




class Node:
	def __init__(self,p,parent):
		self.p = p
		self.parent = parent

def rrt_path_planner_multigoal(room,robot_pose,goal_poses,threshold=1,grid_size=0.25,viz=False):
	
	robot_x = robot_pose[0]
	robot_z = robot_pose[2]
	robot_r = robot_pose[3] #rotation

	random.shuffle(goal_poses)

	print("IMPROTANT:",goal_poses)

	goal_x = goal_poses[0][0]
	goal_z = goal_poses[0][1]

	root = Node(p=np.array([robot_x,robot_z]),parent=None) #The root node

	planning_tree = [root]

	x,y = zip(*room) #convert a bunch of (x,y) into 2 bunches of separated x and y.
	#print(room)

	complete = False
	steps = 0

	while not complete:
		random_steer_goal = random.choice(goal_poses)
		goal_x = random_steer_goal[0]
		goal_z = random_steer_goal[1]
		random_point = random.choice([random.choice(room),[goal_x,goal_z]])
		nearest = nearest_neighbor(random_point,planning_tree)
		steering = steer(random_point,nearest,room,grid_size)
		new_node = Node(steering,nearest)
		planning_tree.append(new_node)
		steps += 1

		'''
		plt.plot(x,y,'bo')
		plt.plot(root.p[0],root.p[1],'ro')
		plt.plot(goal_x,goal_z,'yo')
		plt.plot(steering[0],steering[1],'go')
		plt.show()
		'''
        
		if nearest_neighbor_free(new_node.p,goal_poses)[1] < threshold:
			complete = True
			goal_rep = new_node #the configuration close enough to the goal.

	print("RRT completed in ",steps, " steps")

	plt.plot(x,y,'bo')
	for point in planning_tree:
		plt.plot(point.p[0],point.p[1],'go')
		if point != root:
			x,y = zip(*(point.parent.p,point.p))
			plt.plot(x,y,'b')
            
	child = goal_rep
	parent = goal_rep.parent
	plan_length = 1
	pos_plan = []
	while parent != root:
		pos_plan.append(child.p)
		x,y = zip(*(parent.p,child.p))
		plt.plot(x,y,'r')
    
		child = parent
		parent = child.parent
		plan_length +=1
	pos_plan.append(root.p)
	x,y = zip(*(root.p,child.p))
	plt.plot(x,y,'r')
	print("RRT found a plan of length ",plan_length)
        
	plt.plot(root.p[0],root.p[1],'ro')
	for gp in goal_poses:
		plt.plot(gp[0],gp[1],'yo')
	#plt.plot(random_point[0],random_point[1],'bo')
	#print(robot_x,robot_z)
	#print(goal_x,goal_z)
	#print(pos_plan)
	if viz:
		plt.show()

	return(pos_plan)


def rrt_path_planner(room,robot_pose,goal_pose,threshold=1,grid_size=0.25,viz=False):
	
	robot_x = robot_pose[0]
	robot_z = robot_pose[2]
	robot_r = robot_pose[3] #rotation

	goal_x = goal_pose[0]
	goal_z = goal_pose[2]
	goal_r = goal_pose[3] #rotation

	root = Node(p=np.array([robot_x,robot_z]),parent=None) #The root node

	planning_tree = [root]

	x,y = zip(*room) #convert a bunch of (x,y) into 2 bunches of separated x and y.
	#print(room)

	sp = steer([goal_x,goal_z],root,room)

	complete = False
	steps = 0

	while not complete:
		random_point = random.choice([random.choice(room),[goal_x,goal_z]])
		nearest = nearest_neighbor(random_point,planning_tree)
		steering = steer(random_point,nearest,room,grid_size)
		new_node = Node(steering,nearest)
		planning_tree.append(new_node)
		steps += 1

		'''
		plt.plot(x,y,'bo')
		plt.plot(root.p[0],root.p[1],'ro')
		plt.plot(goal_x,goal_z,'yo')
		plt.plot(steering[0],steering[1],'go')
		plt.show()
		'''
        
		if np.linalg.norm(new_node.p-np.array([goal_x,goal_z])) < threshold:
			complete = True
			goal_rep = new_node #the configuration close enough to the goal.

	#print("RRT completed in ",steps, " steps")

	plt.plot(x,y,'bo')
	for point in planning_tree:
		plt.plot(point.p[0],point.p[1],'go')
		if point != root:
			x,y = zip(*(point.parent.p,point.p))
			plt.plot(x,y,'b')
            
	child = goal_rep
	parent = goal_rep.parent
	plan_length = 1
	pos_plan = []
	while parent != root:
		pos_plan.append(child.p)
		x,y = zip(*(parent.p,child.p))
		plt.plot(x,y,'r')
    
		child = parent
		parent = child.parent
		plan_length +=1
	pos_plan.append(root.p)
	x,y = zip(*(root.p,child.p))
	plt.plot(x,y,'r')
	#print("RRT found a plan of length ",plan_length)
        
	plt.plot(root.p[0],root.p[1],'ro')
	plt.plot(goal_x,goal_z,'yo')
	#plt.plot(random_point[0],random_point[1],'bo')
	#print(robot_x,robot_z)
	#print(goal_x,goal_z)
	#print(pos_plan)
	if viz:
		plt.show()

	return(pos_plan)



if __name__ == "__main__":
	mysave = pickle.load(open("action_maps/FloorPlan2_amap.p","rb"))
	room = mysave["room"]

	clean_room = [[pos_dict['x'],pos_dict['z']] for pos_dict in room]

	action_map = mysave["action_map"]

	goal_pose = [00.06,00.97,-00.17,0] #egg location
	goal_poses = [[00.06,-00.17],[00.31,-00.17],[-01.51,00.97]] #egg location
	robot_pose = [-00.15,01.29,03.70,0] #light switch
	#rrt_path_planner(clean_room,robot_pose,goal_pose)
	rrt_path_planner_multigoal(clean_room,robot_pose,goal_poses)

