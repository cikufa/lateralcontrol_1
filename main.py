import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent
from lateralenv import lateralenv
import xlsxwriter
import warnings
import pandas as pd
import tensorflow as tf
import shapely.geometry as geom

warnings.filterwarnings("ignore")

# if __name__ == '__main__':
discreate_road_pd = pd.read_csv('Chaos-generted road.csv')
road = discreate_road_pd.to_numpy()

agent = Agent(layer1_dim=128, layer2_dim=64, n_actions=2, alpha_A=0.0003, alpha_C=0.005, gamma=0.99)
n_episodes = 100
data_length = int(road.shape[0] / 10)  # 10,000
max_ep_length = 300  # could be int(data_length / n_episodes)
env = lateralenv(road, data_length, n_episodes, max_ep_length)

cnt = 0
dist_limit = 5
ang_limit1 = 0.79;
ang_limit2 = -0.79;  # 45 degree
bad_reward = 10
res = 8
b = 0  # for render
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
for ep in range(1, n_episodes + 1):
    score = 0
    al = [];
    cl = [];
    rewards = []
    # print("b4 reset")
    state, pre_point = env.reset(ep_pointer)  # (1,2)
    # print("after reset state", state)
    states_ = []
    ep_length = 0
    # print(ep, ":____________________________________________________________________________")
    while True:
        action = agent.choose_action(state)
        # assert action != act_buffer , "equal actions !!"
        # act_buffer = action

        newvars, state_, reward, reward_calc, Done, pre_point = env.step(action, ep_length, pre_point, ep_length)

        if Done == 1:
            break
        # if ep_length >100:
        #   break
        else:
            states_.append(state_)
            reward = tf.get_static_value(reward)
            score = score + reward
            rewards.append(reward)

            # if not load_checkpoint:
            closs, aloss, grad1 = agent.learn(state, reward, state_, Done)
            # log
            log.write(ep_pointer + ep_length + 1, 0, f"{ep} / {ep_length}")
            log.write(ep_pointer + ep_length + 1, 3, newvars[0])
            log.write(ep_pointer + ep_length + 1, 4, str(newvars[2:4]))
            log.write(ep_pointer + ep_length + 1, 5, state_[0])
            log.write(ep_pointer + ep_length + 1, 6, state_[1])
            log.write(ep_pointer + ep_length + 1, 7, np.cos(newvars[2] / 200)[0] / 4)
            log.write(ep_pointer + ep_length + 1, 8, newvars[-1])
            log.write(ep_pointer + ep_length + 1, 9, reward)
            log.write(ep_pointer + ep_length + 1, 10, reward_calc)

            state = state_
            ep_length += 1  # step counter

    states_ = np.array(states_)
    score_history.append(score * ep_length / 100)  # ep length should affect score
    ep_pointer += max(int(ep_length / res), 1)

    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
    if (ep % 1 == 0):
        print('episode', ep, 'ep length ', ep_length, 'score', score, 'avg_score', avg_score)
        env.render(ep, score, ep_length)

workbook.close()

if not load_checkpoint:
    ep = [i + 1 for i in range(n_episodes)]
    x = np.array(ep).reshape(n_episodes, 1)
    score_history = np.array(score_history).reshape(n_episodes, 1)
    plt.xlabel("episode")
    plt.ylabel("score")
    plt.plot(x, score_history)
    plt.savefig('scores.png')