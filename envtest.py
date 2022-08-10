import numpy as np
import math
import matplotlib.pyplot as plt
import shapely.geometry as geom
from shapely.ops import nearest_points

class envtest:
    def __init__(self, data):
        # constants
        dt = 0.01
        vx = 10
        iz = 2278.8
        m = 1300
        a1 = 1;
        a2 = 1.5
        caf = 60000
        car = 60000
        cb = -(caf + car);
        cr = (-a1 * caf + a2 * car) / vx
        db = -(a1 * caf - a2 * car);
        dr = -(a1 ** 2 * caf + a2 ** 2 * car) / vx
        cd = caf;
        dd = a1 * caf
        self.constants = [dt, vx, iz, m, cb, cr, db, dr, cd, dd]

        self.road = data[0:data.shape[0], :]
        self.x = data[0:data.shape[0], 0]
        self.y = data[0:data.shape[0], 1]
        self.data_ep = []

        self.max_ep_length = 300
        self.res = 0.1
        self.cnt = 0
        self.dist_limit = 8
        self.ang_limit1 = 0.6;
        self.ang_limit2 = -0.6;  # 45 degree0.1
        self.bad_reward = 10
        self.res = 0.1  # x_acc
        self.b = 0  # for render
        self.load_checkpoint = False
        self.sim_dt = 0.01
        self.preview_dt = 0.03
        self.action_dt = 1
        self.reward_dt = 0.1
        self.learn_dt = 1
        self.t_cnt = 0

        self.Done = 0
        self.coordinates = []
        self.nearestPiontCheck = []

        self.vars = np.zeros((5, 1))
        self.vars_ = np.zeros((5, 1), dtype='float64')  # is only updated for normal step
        self.vars_tmp = np.zeros((5, 1))  # is updated for both normal step and preview step

    def dist_diff(self, limit):  # =geom.Point(0,0)):
        vy, r, x, y, psi = self.vars

        # 3: based on car's vertical disance with the road
        minus = self.data_ep - np.array((x, y)).reshape([1, 2])
        distarr = np.sqrt(minus[:, 0] ** 2 + minus[:, 1] ** 2)
        ind = np.argmin(distarr)

        ## ang
        road_slope_rad = np.arctan2((self.data_ep[ind + 1, 1] - self.data_ep[ind, 1]),
                                    (self.data_ep[ind + 1, 0] - self.data_ep[ind, 0])) if ind != len(
            self.data_ep) - 1 else np.arctan2((self.data_ep[ind, 1] - self.data_ep[ind - 1, 1]),
                                              (self.data_ep[ind, 0] - self.data_ep[ind - 1, 0]))
        angle_diff = abs(road_slope_rad - psi)[0]
        limited_angle_diff = max(angle_diff, 0.005)

        ## dist
        dist = (((self.data_ep[ind, 0] - x) ** 2 + (self.data_ep[ind, 1] - y) ** 2) ** 0.5)[0]
        limited_dist = max(dist, 0.01)  # for makhraj problems

        # print("dist_diff: ", x, y)

        if limit == 1:
            return limited_dist, limited_angle_diff
        else:
            return dist, angle_diff

    def sim_step(self, action):
        dt, vx, iz, m, cb, cr, db, dr, cd, dd = self.constants
        vy, r, x, y, psi = np.vsplit(self.vars, 5)
        dt = self.sim_dt
        self.t_cnt += 1
        # calc new state
        par_mat1 = np.array([[cb / (m * vx), cr / m - vx, 0, 0, 0],
                             [db / (iz * vx), dr / iz, 0, 0, 0],
                             [-math.sin(psi), 0, 0, 0, 0],
                             [math.cos(psi), 0, 0, 0, 0],
                             [0, 1, 0, 0, 0]])

        par_mat2 = np.array([[cd * action / m], [dd * action / iz], [vx * math.cos(psi)],
                             [vx * math.sin(psi)], [0]], dtype='float64')

        var_dot_mat = par_mat1 @ self.vars + par_mat2  # (5,1)= (5,5)@(5,1)+(5,1)

        self.vars = self.vars + dt * var_dot_mat  # (5,1) =(5,1)+(5,1)
        self.coordinates.append(self.vars[2:4, 0])

    def normalize(self, d, a):
        # return d/(dist_limit-0), a/(ang_limit1-ang_limit2)
        return d / (self.dist_limit - 0), a

    def calc_reward(self, dist, angle_diff, action, ep_length):
        weight = 0.01
        action_weight = -0.01
        preview_weight = 0.001
        k1 = 1 / dist + 1 / angle_diff ** 2
        #k2 = 1 / future_dist + 1 / future_angle_diff ** 2
        ep_len_weight = 1
        reward_calc = f'{weight} * {k1} + {action_weight} * {action} + {ep_len_weight} * {ep_length}'
        reward = k1 * weight + action_weight * action + ep_len_weight * (
                self.max_ep_length - self.episode_length_cnt)
        return reward, reward_calc

    def step(self, action, ep_length):  # handle done,
        self.sim_step(action)
        dist, angle_diff = self.dist_diff(limit=1)

        ## debug only
        if ep_length == 0 and angle_diff > self.ang_limit1:
            dist, angle_diff= self.dist_diff(limit=1)
        # --------------------------------------------

        self.episode_length_cnt = self.episode_length_cnt - 1 #???

        if dist > self.dist_limit or angle_diff > self.ang_limit1 or angle_diff < self.ang_limit2 or self.episode_length_cnt == 0:
            print("last dist", dist, "last angle", angle_diff)
            self.Done = 1
            return self.bad_reward, 'nothing'
        else:
            dist, angle_diff = self.normalize(d=dist, a=angle_diff)
            reward, reward_calc = self.calc_reward(dist, angle_diff, action, ep_length)
            self.state_ = np.array([dist, angle_diff]).reshape((1,2))  # real state (not limited)

            return reward, reward_calc  # state:(dist, ang_dif)

    def reset(self, ep_pointer):  # before each episode
        if ep_pointer > (self.road.shape[0] - 300):  # =max_episode_length. resets the road and stars from the begining
            ep_pointer = 0

        self.Done = 0
        self.episode_length_cnt = self.max_ep_length
        self.coordinates = []

        # a new section of the road excel is selected for each episode
        self.data_ep = self.road[ep_pointer:ep_pointer + int(self.max_ep_length / self.res), :]

        # the car starts on the road
        st_vy = 0
        st_r = 0
        st_x = self.data_ep[0, 0]  # + np.random.rand()
        st_y = self.data_ep[0, 1]  # + np.random.rand()
        st_psi = np.arctan2((self.data_ep[1, 1] - self.data_ep[0, 1]), (self.data_ep[1, 0] - self.data_ep[0, 0]))

        self.coordinates.append([st_x, st_y])

        self.vars = np.array([[st_vy, st_r, st_x, st_y, st_psi]], dtype='float64').T

        limited_dist0, limited_angle_diff0 = self.dist_diff(limit=1)
        limited_dist0, limited_angle_diff0 = self.normalize(limited_dist0, limited_angle_diff0)

        state0_ep = np.array([limited_dist0, limited_angle_diff0]).reshape((1, 2))  # (1,2)

        return state0_ep, ep_pointer

    def render(self, ep, score, ep_length, pnt, alosses, closses):
        ### f1 = road and path
        plt.figure(1)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.plot(self.data_ep[:, 0], self.data_ep[:, 1], 'r')  # road

        if ep_length != 0:
            plt.plot(np.array(self.coordinates)[:, 0], np.array(self.coordinates)[:, 1], label=score)  # path

        if ep % 10 == 0 and ep_length != 0:
            plt.legend()
            plt.savefig(f"paths/path{ep}.jpg")
            plt.cla()
            b = 0

