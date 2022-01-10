import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from ai2thor.controller import Controller
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DQN, self).__init__()
        self.fcn1 = nn.Linear(in_dim,hidden_dim)
        self.fcn2 = nn.Linear(hidden_dim,hidden_dim)
        self.fcn3 = nn.Linear(hidden_dim,out_dim)


    def forward(self, x):
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = self.fcn3(x)
        return x

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

gridSize = 0.25
controller = Controller(scene="FloorPlan1", gridSize=gridSize,width=500,height=500,renderObjectImage=False,renderClassImage=False,renderDepthImage=False,snapToGrid=False)

# Get number of actions from gym action space
n_actions = 3 #0,45,90,...,315 and move forward
n_input = 6 #2d pos and 1d rot of robot, 2d pos and 1d rot of object
n_hidden = 16 #size of hidden dimensions

policy_net = DQN(n_input, n_hidden, n_actions)
policy_net.double()
target_net = DQN(n_input, n_hidden, n_actions)
target_net.double()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.stack(batch.reward)
    reward_batch = reward_batch.squeeze(1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = policy_net(state_batch)
    state_action_values = state_action_values.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).float().max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def take_env_step(action, obj_ref):
    if action == 0:
        event = controller.step(
            action="RotateRight",
            degrees=45)
    elif action == 1:
        event = controller.step(
        action="RotateRight",
        degrees=90)
    elif action == 2:
        event = controller.step(
        action="RotateRight",
        degrees=135)
    elif action == 3:
        event = controller.step(
            action="RotateRight",
            degrees=180)
    elif action == 4:
        event = controller.step(
            action="RotateRight",
            degrees=225)
    elif action == 5:
        event = controller.step(
            action="RotateRight",
            degrees=270)
    elif action == 6:
        event = controller.step(
            action="RotateRight",
            degrees=315)
    elif action == 7:
        event = controller.step(
            action="MoveAhead",
            moveMagnitude=0.25)


def take_env_step_simple(action, obj_ref):
    if action == 0:
        event = controller.step(
            action="RotateRight",
            degrees=45)
    elif action == 1:
        event = controller.step(
        action="RotateLeft",
        degrees=45)
    elif action == 2:
        event = controller.step(
            action="MoveAhead",
            moveMagnitude=0.25)

    #check if at end of task
    try:
        event = controller.step(action='PickupObject',
            objectId=obj_ref['objectId'],
            raise_for_failure=True)
        #succeeded, give reward +1 and done
        return None, 1, True, None
    except:
        #failed, keep navigating
        return None, 0, False, None

def get_state():
    event = controller.step(action = "AdvancePhysicsStep", timeStep=0.01)

    agent = event.metadata['agent']
    global_robot_loc = agent['position']
    global_robot_rot = agent['rotation']

    obj = "Mug"
    all_objs = event.metadata['objects']
    for possible_obj in all_objs:
        #get an object of right category but not assigned yet
        if obj in possible_obj['objectId']:
            obj_ref = possible_obj
            break

    #object loc
    obj_loc = obj_ref['position']
    obj_rot = obj_ref['rotation']


    state = torch.from_numpy(np.array([global_robot_loc['x'],
        global_robot_loc['z'],
        global_robot_rot['y'],
        obj_loc['x'],
        obj_loc['z'],
        obj_rot['y']])).double()

    return obj_ref, state

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    controller.reset(scene="FloorPlan1")

    obj_ref, state = get_state()

    finished = False

    for t in range(50):#count():
        # Select and perform an action
        action = select_action(torch.unsqueeze(state.double(),0))
        _, reward, done, _ = take_env_step_simple(action.item(), obj_ref)
        reward = torch.tensor([reward])

        # Observe new state
        if not done:
            obj_ref, next_state = get_state()
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            finished = True
            episode_durations.append(t + 1)
            print("episode durations")
            print(episode_durations)
            print(EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY))
            print("=============")
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if not finished:
        episode_durations.append(t)

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()