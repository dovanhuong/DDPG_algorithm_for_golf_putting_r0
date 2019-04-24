#!/usr/bin/env python
#################################################
#             Authors: Huong Do Van             #
#         Email: huong.dovan@kist.re.kr         #
#   Korea Institute of Science and Technology   #
#################################################
import gym
import gym.spaces
from ur_gazebo_test2.scripts import slide_puck
import rospy
import numpy as np
import time
import universal_robot.ur_kinematics.src.ur_kin_py as ur_kin_py
import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32
import torch
import torch.nn.functional as F
import gc
import torch.nn as nn
import math
from collections import deque
import csv

# ---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))

# Write data to .csv file:
def write_csv(data):
    with open('Training_result_01.csv', 'a') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(data)

# ---Functions to make network updates---#

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# ---Ornstein-Uhlenbeck Noise for action---#

class ActionNoise:
    # Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        print('aqu2i' + str(self.X))
        return self.X



#---Critic--#

EPS = 0.003
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1./np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v,v)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fa1 = nn.Linear(action_dim, 256)
        self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())

        self.fca1 = nn.Linear(512, 512)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fca2 = nn.Linear(512, 1)
        self.fca2.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action):
        xs = torch.relu(self.fc1(state))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs


# ---Actor---#

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit_l1, action_limit_l2, action_limit_theta1, action_limit_theta2):

        super(Actor, self).__init__()
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_l1 = action_limit_l1
        self.action_limit_l2 = action_limit_l2
        self.action_limit_theta1 = action_limit_theta1
        self.action_limit_theta2 = action_limit_theta2

        self.fa1 = nn.Linear(state_dim, 512)
        self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())

        self.fa2 = nn.Linear(512, 512)
        self.fa2.weight.data = fanin_init(self.fa2.weight.data.size())

        self.fa3 = nn.Linear(512, action_dim)
        self.fa3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        action = self.fa3(x)
        if state.shape == torch.Size([2]):
            action[0] = torch.tanh(action[0]) * self.action_limit_l1
            action[1] = torch.tanh(action[1]) * self.action_limit_l2
            action[2] = torch.tanh(action[2]) * self.action_limit_theta1
            action[3] = torch.tanh(action[3]) * self.action_limit_theta2
        else:
            action[:, 0] = torch.tanh(action[:, 0]) * self.action_limit_l1
            action[:, 1] = torch.tanh(action[:, 1]) * self.action_limit_l2
            action[:, 2] = torch.tanh(action[:, 2]) * self.action_limit_theta1
            action[:, 3] = torch.tanh(action[:, 3]) * self.action_limit_theta2
        return action


# ---Memory Buffer---#

class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])

        return s_array, a_array, r_array, new_s_array

    def len(self):
        return self.len

    def add(self, s, a, r, new_s):
        transition = (s, a, r, new_s)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)


#---Where the train is made---#

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001


class Trainer_play_golf:
    def __init__(self, state_dim, action_dim, action_limit_l1, action_limit_l2, action_limit_theta1, action_limit_theta2, ram):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_l1 = action_limit_l1
        self.action_limit_l2 = action_limit_l2
        self.action_limit_theta1 = action_limit_theta1
        self.action_limit_theta2 = action_limit_theta2
        self.ram = ram
        self.noise = ActionNoise(self.action_dim)

        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit_l1, self.action_limit_l2, self.action_limit_theta1, self.action_limit_theta2)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_limit_l1, self.action_limit_l2, self.action_limit_theta1, self.action_limit_theta2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)
        self.pub_qvalue = rospy.Publisher('qvalue', Float32, queue_size=5)
        self.qvalue = Float32()

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        state = torch.from_numpy(state)
        action = self.target_actor.forward(state).detach()
        # print('actionploi', action)
        return action.data.numpy()

    def get_exploration_action(self, state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        # noise = self.noise.sample()
        # print('noise', noise)
        new_action = action.data.numpy()  # + noise
        # print('action_no', new_action)
        return new_action

    def optimizer(self):
        s_sample, a_sample, r_sample, new_s_sample = ram.sample(BATCH_SIZE)

        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)

        # -------------- optimize critic

        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
        # y_exp = r _ gamma*Q'(s', P'(s'))
        y_expected = r_sample + GAMMA * next_value
        # y_pred = Q(s,a)
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))
        # -------Publisher of Vs------
        self.qvalue = y_predicted.detach()
        self.pub_qvalue.publish(torch.max(self.qvalue))
        # print(self.qvalue, torch.max(self.qvalue))
        # ----------------------------
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ------------ optimize actor
        pred_a_sample = self.actor.forward(s_sample)
        loss_actor = -1 * torch.sum(self.critic.forward(s_sample, pred_a_sample))

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)

    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), dirPath + '/Models/' + str(episode_count) + '_actor1.pt')
        torch.save(self.target_critic.state_dict(), dirPath + '/Models/' + str(episode_count) + '_critic1.pt')
        print('****Models saved***')

    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(dirPath + '/Models/' + str(episode) + '_actor1.pt'))
        self.critic.load_state_dict(torch.load(dirPath + '/Models/' + str(episode) + '_critic1.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('***Models load***')


#---Run agent---#

is_training = True # True for exploration

if is_training:
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.05
else:
    exploration_rate = 0.05
    max_exploration_rate = 0.05
    min_exploration_rate = 0.05

exploration_decay_rate = 0.001

MAX_EPISODES = 10001 #10001
MAX_STEPS = 1
MAX_BUFFER = 100000
rewards_all_episodes = []

STATE_DIMENSION = 2
ACTION_DIMENSION = 4


ACTION_L1_MIN = 0.30
ACTION_L2_MIN = 0.15
ACTION_THETA1_MIN = -0.08
ACTION_THETA2_MIN = -0.08

ACTION_L1_MAX = 0.35
ACTION_L2_MAX = 0.20
ACTION_THETA1_MAX = 0.08
ACTION_THETA2_MAX = 0.08


if is_training:
    var_l1 = ACTION_L1_MAX
    var_l2 = ACTION_L2_MAX
    var_theta1 = ACTION_THETA1_MAX
    var_theta2 = ACTION_THETA2_MAX

else:
    var_l1 = ACTION_L1_MAX*0.1
    var_l2 = ACTION_L2_MAX*0.1
    var_theta1 = ACTION_THETA1_MAX*0.1
    var_theta2 = ACTION_THETA2_MAX*0.1

print('State Dimensions: ' + str(STATE_DIMENSION))
print('Action Dimensions: ' + str(ACTION_DIMENSION))
print('Action set Max: l1 = ' + str(ACTION_L1_MAX) + '  l2 = ' + str(ACTION_L2_MAX) + '(mm) and theta1 = ' + str(ACTION_THETA1_MAX) + '  theta2 =' + str(ACTION_THETA2_MAX) + '(rad)')

ram = MemoryBuffer(MAX_BUFFER)
trainer = Trainer_play_golf(STATE_DIMENSION, ACTION_DIMENSION, ACTION_L1_MAX, ACTION_L2_MAX, ACTION_THETA1_MAX, ACTION_THETA2_MAX, ram)
trainer.load_models(9650)


if __name__ == '__main__':

    rospy.init_node('UR5_play_golf')
    rospy.Time.now()

    env = gym.make('UR5Slide-v1')
    env.env.my_init(True)   # True: use gazebo sim,  False: use skkim's sim
    rospy.loginfo("gym env done")

    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()

    start_time = time.time()
    past_action = np.array([0.30, 0.15, -0.08, -0.08])
    csvData = []
    num_success = 0

    with open('Training_result_01.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        csvData = [['initialposition_x', 'initialposition_y', 'achieved_position_x', 'achieved_position_y',
                    'success']]
        writer.writerows(csvData)
    csvFile.close()


    for ep in range(MAX_EPISODES):

        done = False
        state = env.reset()

        print('Episode: ' + str(ep))
        rewards_current_episode = 0.0

        for step in range(MAX_STEPS):
            state = np.float32(state)
            initialposition_x = state[0]
            initialposition_y = state[1]
            if is_training:
                action = trainer.get_exploration_action(state)

                action[0] = np.clip(np.random.normal(action[0], var_l1),ACTION_L1_MIN,ACTION_L1_MAX)
                action[1] = np.clip(np.random.normal(action[1], var_l2), ACTION_L2_MIN, ACTION_L2_MAX)
                action[2] = np.clip(np.random.normal(action[2], var_theta1), ACTION_THETA1_MIN, ACTION_THETA1_MAX)
                action[3] = np.clip(np.random.normal(action[3], var_theta2), ACTION_THETA2_MIN, ACTION_THETA2_MAX)
                #action = [action[0], action[1], action[2],action[3]]
                #env.step(action)

            if not is_training:
                action = trainer.get_exploitation_action(state)

            next_state, reward, done, info = env.step(action)
            achieved_position_x = next_state[0]
            achieved_position_y = next_state[1]

            if reward == 100:
                success = 1
                num_success += 1
            else:
                success = 0

            past_action = action
            csvData = [initialposition_x, initialposition_y, achieved_position_x, achieved_position_y, success]

            with open('Training_result_01.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                csvData = [[str(initialposition_x), str(initialposition_y), str(achieved_position_x), str(achieved_position_y), str(success)]]
                writer.writerows(csvData)

            csvFile.close()

            rewards_current_episode += reward
            next_state = np.float32(next_state)

            ram.add(state, action, reward, next_state)
            state = next_state

            if ram.len >= 0*MAX_STEPS and is_training:
                var_l1 = max([var_l1*0.9999, 0.30*ACTION_L1_MAX])
                var_l2 = max([var_l2*0.9999, 0.30*ACTION_L2_MAX])
                var_theta1 = max([var_theta1*0.9999, 0.30*ACTION_THETA1_MAX])
                var_theta2 = max([var_theta2*0.9999, 0.30*ACTION_THETA2_MAX])
                trainer.optimizer()


            if done or step == MAX_STEPS-1:
                print(' Reward per ep: ' + str(reward))
                print('Explore_l1: ' + str(var_l1) + ' Explore_l2: ' + str(var_l2) + ' Explore_theta1: ' + str(var_theta1) + ' Explore_theta2: ' + str(var_theta2))
                rewards_all_episodes.append(reward)
                result = reward
                pub_result.publish(result)

                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                break
        csvData.append(csvData)
        now_sucess_rate = num_success/(ep+1)*100.00
        print('The percentage over', ep, 'is: ', now_sucess_rate, '(%)')

    exploration_rate = (min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*ep))
    gc.collect()

    if ep%50 == 0:
        trainer.save_models(ep)

print('Completed Training')

