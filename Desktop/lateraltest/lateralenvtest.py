import random

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
from shapely.ops import nearest_points
import math


class lateralenv:
    def __init__(self, n_episodes, max_ep_length):
        # constants
        self.n_episodes = n_episodes
        self.episode_length_cnt = max_ep_length
        self.max_ep_length = max_ep_length  # could be int(data_length / n_episodes)
        self.score = 0
        self.index = 0
        self.Done = 0
        self.coordinates = []
       
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.dist_limit1 = 200 
        self.dist_limit2= -200
        self.bad_reward = -1000
        self.sim_dt = 0.01
        self.reward_dt = 0.1
        self.learn_dt = 1
        self.action_dt = 1
        self.t_cnt = 0
        self.loc= 0

    def sim_step(self, action):
        self.loc= self.loc + action
        self.t_cnt = self.t_cnt + 1 
        

    def normalize(self, d, a):
        # return d/(dist_limit-0), a/(ang_limit1-ang_limit2)
        return d / (self.dist_limit - 0), a

    def calc_reward(self, ep_length):
        reward = 1 /np.abs(self.loc) 
        # reward = - np.abs(self.loc)
        return reward

    def step(self, ep_length):  # handle done,
        
        self.episode_length_cnt = self.episode_length_cnt - 1

        if self.loc > self.dist_limit1 or self.loc < self.dist_limit2:
            self.Done = 1
            print("last loc", self.loc)
            return self.bad_reward

        elif self.episode_length_cnt == 0:
            self.Done = 1
            return 0

        else:
            # self.loc = self.normalize(self.loc)
            reward = self.calc_reward(ep_length)
            self.state_ = np.array([self.loc]).reshape((1,1))  # real state (not limited)
            
        return reward  # state:(dist, ang_dif)
       

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # def render(self, ep, score, ep_length, pnt, alosses, closses):
    #     ### f1 = road and path
    #     plt.figure(1)
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.plot(self.data_ep[:, 0], self.data_ep[:, 1], 'r')  # road
    #     if ep_length != 0:
    #         plt.plot(np.array(self.coordinates)[:, 0], np.array(self.coordinates)[:, 1], label=score)  # path
    #     if ep % 10 == 0 and ep_length != 0:
    #         plt.legend()
    #         plt.savefig(f"paths/path{ep}.jpg")
    #         plt.cla()
    #         b = 0

    #     #if ep_length != 0:
    #     #    plt.plot(np.array(self.coordinates)[:, 0], np.array(self.coordinates)[:, 1], label=score)  # path

    #     ### f2 = aloss

    #     if alosses != []:
    #         plt.figure(2)
    #         xa = np.arange(len(alosses))
    #         plt.plot(xa, np.array(alosses)[:, 0, 0])
    #         plt.savefig(f"aloss/aloss{ep}.jpg")
    #         plt.cla()

    #     # ### f3 = closs

    #     if closses != []:
    #         plt.figure(3)
    #         xc = np.arange(len(closses))
    #         plt.plot(xc, np.array(closses)[:, 0, 0])
    #         plt.savefig(f"closs/closs{ep}.jpg")
    #         plt.cla()


    def reset(self):  # before each episode
        self.Done = 0
        self.episode_length_cnt = self.max_ep_length
        self.coordinates = []

        self.loc = 1

        self.coordinates.append([self.loc])
        
        self.state = np.array([self.loc]).reshape((1, 1))  # (1,2)
