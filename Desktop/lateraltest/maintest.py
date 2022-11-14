import numpy as np
import matplotlib.pyplot as plt
from Agenttest import Agent
from lateralenvtest import lateralenv
import xlsxwriter
import warnings
import pandas as pd
import tensorflow as tf
import traceback
import keras 
#import torch
warnings.filterwarnings("ignore")
#device = torch.device("cpu:0")
import math
if __name__ == '__main__':
    agent = Agent(layer1_dim=128, layer2_dim=64, n_actions=2, alpha_A=0.0003, alpha_C=0.005, gamma=0.5)
    n_episodes = 5001
    max_ep_length = 150 # could be int(data_length / n_episodes)
    env = lateralenv(n_episodes, max_ep_length)

    cnt = 0
    res = 0.1 #0.1
    score_history = []
    best_score = 0  # reward = 1/positive > 0 -> min score =0
    load_checkpoint = False
    checkpoints = [2000, 5000]

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

    epnum = 1
    
    try:
        for ep in range(1, n_episodes + 1):
            # plt.show()
            # plt.savefig(f'moves/ep{ep}')
            epnum = ep
            alosses = []
            closses = []
            score = 0
            al = [];
            cl = [];
            rewards = []
            
            env.reset()  # self.state=0
            action = agent.choose_action(env.state)  #initial action t=n
            states_ = []
            ep_length = 1
            reward_for_few_steps = 0
            Done = 0
            reward = 0
            
            while True:
                env.sim_step(action)
                #print("t_cnt : ", env.t_cnt)
                if env.t_cnt % (env.reward_dt/env.sim_dt) == 0:   # t=10n
                    reward = env.step(ep_length) #env.state_ , env.Done
                    # print("reward", reward)
                    reward_for_few_steps += reward
                    ep_length += 1 # step counter

                if env.Done == 1:
                    break

                if env.t_cnt % (env.learn_dt/env.sim_dt) == 0:  #t=100n
                    states_.append(env.state_)
                    reward_for_few_steps = tf.get_static_value(reward_for_few_steps)
                    score = score + reward_for_few_steps
                    # print(score)
                    rewards.append(reward_for_few_steps)

                    # if not load_checkpoint:
                    closs, aloss, grad1 = agent.learn(env.state, reward_for_few_steps, env.state_, env.Done)
                    # alosses.append(aloss)
                    # closses.append(closs)
                    reward_for_few_steps = 0

                if env.t_cnt % (env.action_dt/env.sim_dt) == 0:
                    action = agent.choose_action(env.state)
                    env.state = env.state_

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
            #score_history.append(score * math.sqrt(ep_length))  # ep length should affect score
            score_history.append(score)
            # plt.plot(road[ep_pointer, 0], road[ep_pointer, 1], 'bo')
            # plt.show()
       
            avg_score = np.mean(score_history[-100:])
            if avg_score > best_score:
                best_score = avg_score

            print('episode', ep, 'ep length ', ep_length, 'score', score, 'avg_score', avg_score)
            
            # env.render(ep, score * ep_length / 100, ep_length, ep_pointer, alosses, closses)

            if ep in checkpoints:
                agent.actor.save('model')
               

    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        # print(agent.actor.summary())
        # print('bbbbbbbbbbbbbbbbbb')
        workbook.close()
        if not load_checkpoint:
            score_history = np.array(score_history).reshape((-1,1,1))
            print(score_history.shape)
            x = np.arange(0,score_history.shape[0])
            print(x.shape)
            pltlen = int(n_episodes//10)

            plt.xlabel("episode")
            plt.ylabel("score")
            plt.plot(x[:], score_history[:, 0, 0])
            plt.savefig(f'scores.png')
            plt.cla()
        
        else:
            np.load(checkpoints);

