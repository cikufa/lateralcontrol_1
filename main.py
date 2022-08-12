import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent
from lateralenv import lateralenv
import xlsxwriter
import warnings
import pandas as pd
import tensorflow as tf
#import torch
warnings.filterwarnings("ignore")
#device = torch.device("cpu:0")

def my_round(x, num_decimal_precision_digits):
    power_of_10 = 10 ** num_decimal_precision_digits
    return round(x * power_of_10)  / power_of_10

if __name__ == '__main__':

    # discreate_road_pd = pd.read_csv('0.1road.csv')
    discreate_road_pd = pd.read_excel('sin road.xlsx')
    road = discreate_road_pd.to_numpy()
    # road= road[:, 1:3]

    agent = Agent(layer1_dim=128, layer2_dim=64, n_actions=2, alpha_A=0.0003, alpha_C=0.005, gamma=0.5)
    n_episodes = 2000
    data_length = int(road.shape[0])  # sin road = 10,000
    max_ep_length = 300  # could be int(data_length / n_episodes)
    env = lateralenv(road, data_length, n_episodes, max_ep_length)

    cnt = 0
    res = 0.1 #0.1
    score_history = []
    best_score = 0  # reward = 1/positive > 0 -> min score =0
    load_checkpoint = False

    workbook = xlsxwriter.Workbook('log.xlsx')
    log = workbook.add_worksheet("ep_per_ep")
    log.write(0, 0, "ep / step")
    log.write(0, 3, "vy")
    log.write(0, 4, "point")
    log.write(0, 5, "distance")
    log.write(0, 6, "angle_diff")
    log.write(0, 7, "road derivative")
    log.write(0, 8, "psi")
    log.write(0, 9, "reward")
    log.write(0, 10, "point dist_diff +  preview point dist_diff + action")

    # training________________________________________________________________________________________
    ep_pointer = 0
    epnum = 1
    
    try:
        for ep in range(1, n_episodes + 1):
            epnum = ep
            alosses = []
            closses = []
            score = 0
            al = [];
            cl = [];
            rewards = []
            
            state, ep_pointer = env.reset(ep_pointer)  # (1,2)
            action = agent.choose_action(state)  # initial action
            states_ = []
            ep_length = 0
            reward_for_few_steps = 0
            Done = 0
            reward = 0
            reward_calc = 0
            while True:
                env.sim_step(action)

                if env.t_cnt % (env.reward_dt/env.sim_dt) == 0:
                    reward, reward_calc = env.step(action, ep_length)
                    reward_for_few_steps += reward
                    ep_length += 1 # step counter

                if env.Done == 1:
                    ep_pointer += 10
                    break

                if env.t_cnt % (env.learn_dt/env.sim_dt) == 0:
                    states_.append(env.state_)
                    reward_for_few_steps = tf.get_static_value(reward_for_few_steps)
                    score = score + reward_for_few_steps
                    rewards.append(reward_for_few_steps)

                    # if not load_checkpoint:
                    closs, aloss, grad1 = agent.learn(state, reward_for_few_steps, env.state_, env.Done)
                    alosses.append(aloss)
                    closses.append(closs)
                    reward_for_few_steps = 0

                if env.t_cnt % (env.action_dt/env.sim_dt) == 0:
                    action = agent.choose_action(state)
                    state = env.state_
                
                # log
                # log.write(ep_pointer + ep_length + 1, 0, f"{ep} / {ep_length}")
                # log.write(ep_pointer + ep_length + 1, 3, newvars[0])
                # log.write(ep_pointer + ep_length + 1, 4, str(newvars[2:4]))
                # log.write(ep_pointer + ep_length + 1, 5, state_[0])
                # log.write(ep_pointer + ep_length + 1, 6, state_[1])
                # log.write(ep_pointer + ep_length + 1, 7, np.cos(newvars[2] / 200)[0] / 4)
                # log.write(ep_pointer + ep_length + 1, 8, newvars[-1])
                # log.write(ep_pointer + ep_length + 1, 9, reward)
                # log.write(ep_pointer + ep_length + 1, 10, reward_calc)
                
            states_ = np.array(states_)
            score_history.append(score * ep_length / 100)  # ep length should affect score
            ep_pointer += max(int(ep_length / res), 1)
            # plt.plot(road[ep_pointer, 0], road[ep_pointer, 1], 'bo')
            # plt.show()
            avg_score = np.mean(score_history[-100:])
            if avg_score > best_score:
                best_score = avg_score

            print('episode', ep, 'ep length ', ep_length, 'score', score, 'avg_score', avg_score)
            if (ep_length %1 == 0):
                env.render(ep, score, ep_length, ep_pointer, alosses, closses)

    finally:

        workbook.close()

        if not load_checkpoint:
            #ep = [i + 1 for i in range(n_episodes)]
            x = np.arange(0,len(score_history)).reshape(-1, 1)
            score_history = np.array(score_history).reshape(-1, 1)
            pltlen = int(n_episodes//1000)
            for i in range(0, x.shape[0], pltlen):
                plt.xlabel("episode")
                plt.ylabel("score")
                plt.plot(x[i:i+pltlen], score_history[i:i+pltlen])
                plt.savefig(f'scores/score{i}.png')
                plt.cla()
