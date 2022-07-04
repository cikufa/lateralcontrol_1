import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
from shapely.ops import nearest_points
import math


class lateralenv:
    def __init__(self, data, data_length, n_episodes, max_ep_length):
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

        self.data_length = data_length
        self.n_episodes = n_episodes
        self.episode_length_cnt = max_ep_length
        self.max_ep_length = max_ep_length  # could be int(data_length / n_episodes)

        self.road = data[0:data_length, :]
        self.x = data[0:data_length, 0]
        self.y = data[0:data_length, 1]
        self.data_ep = []

        self.heading_angle = [np.arctan2(self.y[i + 1] - self.y[i], self.x[i + 1] - self.x[i]) for i in
                              range(self.data_length - 1)]  # rad [-1.57, 1.57]
        self.heading_angle.insert(0, self.heading_angle[0])  # append last value to adjust the shape
        self.heading_angle = np.asfarray(self.heading_angle).reshape(self.data_length, 1)

        # ______________________________________________init vars_____________________________________________________________

        self.score = 0
        self.index = 0
        self.Done = 0
        self.coordinates = []
        self.nearestPiontCheck = []
        self.vys = []
        self.vymax = -10
        self.vars = np.zeros((5, 1))
        self.vars_ = np.zeros((5, 1), dtype='float64')  # is only updated for normal step
        self.vars_tmp = np.zeros((5, 1))  # is updated for both normal step and preview step

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.cnt = 0
        self.dist_limit = 5
        self.ang_limit1 = 0.6;
        self.ang_limit2 = -0.6;  # 45 degree
        self.bad_reward = 10
        self.res = 0.1 #x_acc
        self.b = 0  # for render
        self.load_checkpoint = False

    def dist_diff(self, ep, limit_dist, limit_ang, stp, pre_point):  # =geom.Point(0,0)):
        vy, r, x, y, psi = self.vars_tmp

        # 3: based on car's vertical disance with the road
        minus = self.data_ep - np.array((x, y)).reshape([1,2])
        distarr = np.sqrt(minus[:, 0] ** 2 + minus[:, 1] ** 2)
        ind = np.argmin(distarr)
        #dist = np.min(distarr)
        road_slope = (self.data_ep[ind+1, 1] - self.data_ep[ind, 1])/(self.data_ep[ind+1, 0] - self.data_ep[ind, 0]) if ind!=len(self.data_ep)-1 else (self.data_ep[ind, 1] - self.data_ep[ind-1, 1])/(self.data_ep[ind, 0] - self.data_ep[ind-1, 0])

        point = geom.Point(x, y)
        dist = point.distance(self.road_ep)
        limited_dist = max(dist, 0.01)  # for makhraj problems

        # limited_dist = min(limited_dist, 100)
        # dist_z = math.sqrt((y - self.y0) ** 2 + (x - self.x0) ** 2)

        nearestP = nearest_points(self.road_ep, point)[0]
        self.nearestPiontCheck.append(np.array(nearestP))
        self.nearestPiontCheck.append(np.array(point))
        #road_slope = (nearestP.y - pre_point.y) / (nearestP.x - pre_point.x) if (nearestP.x - pre_point.x) != 0 else (nearestP.y - pre_point.y) / 0.001
        angle_diff = abs(np.arctan2((road_slope - psi), 1))[0]
        # angle_diff = abs(np.arctan2((nearestP.y-pre_point.y),nearestP.x-pre_point.x)- psi[0]) #sara

        # index, = np.where(self.road_ep == nearestP)
        # print("index", index)
        # angle_diff=  np.arctan2(self.road_ep[index+1][1]-self.road_ep[index][1], self.road_ep[index+1][0]- self.road_ep[index][0]) - psi
        # angle_diff = abs((np.cos(x / 100) / 4 - psi)[0])
        limited_angle_diff = max(angle_diff, 0.005)
        # limited_angle_diff=min(limited_angle_diff , 100) #-> max reward = 5000, min reward 5e-5

        if limit_dist == 1:
            if limit_ang == 1:
                return limited_dist, limited_angle_diff, nearestP
            elif limit_ang == 0:
                return limited_dist, angle_diff, nearestP
        else:
            return dist, angle_diff, nearestP

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def preview(self, action,
                preview):  # in this version the preview point is calculated using the updated self.vars. also try with non-updated vars
        dt, vx, iz, m, cb, cr, db, dr, cd, dd = self.constants
        vy, r, x, y, psi = np.vsplit(self.vars, 5)
        if preview == 1:
            dt = 0.3
        if preview == 0:
            dt = 0.1
        # calc new state
        par_mat1 = np.array([[cb / (m * vx), cr / m - vx, 0, 0, 0],
                             [db / (iz * vx), dr / iz, 0, 0, 0],
                             [-math.sin(psi), 0, 0, 0, 0],
                             [math.cos(psi), 0, 0, 0, 0],
                             [0, 1, 0, 0, 0]])

        par_mat2 = np.array([[cd * action / m], [dd * action / iz], [vx * math.cos(psi)],
                             [vx * math.sin(psi)], [0]], dtype='float64')

        var_dot_mat = par_mat1 @ self.vars + par_mat2  # (5,1)= (5,5)@(5,1)+(5,1)

        self.vars_tmp = self.vars + dt * var_dot_mat  # (5,1) =(5,1)+(5,1)
        self.vars_tmp[4, 0] = self.vars[4, 0] + dt * self.vars_tmp[1, 0]
        if preview == 0:
            self.vars_ = self.vars_tmp
        return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def normalize(self, d, a):
        # return d/(dist_limit-0), a/(ang_limit1-ang_limit2)
        return d / (self.dist_limit - 0), a

    def step(self, action, stp_cnt, pre_point, ep_length):
        self.preview(action, preview=0)
        # print("____________________________NOT preview_________________________")
        dist, angle_diff, pre_point2 = lateralenv.dist_diff(self, ep=0, limit_dist=1, limit_ang=0, stp=stp_cnt,pre_point=pre_point)
        #print("dist", dist,"angle diff", angle_diff)
        if ep_length == 0 and angle_diff> self.ang_limit1:
            dist, angle_diff, pre_point2 = lateralenv.dist_diff(self, ep=0, limit_dist=1, limit_ang=0, stp=stp_cnt,
                                                                pre_point=pre_point)
        pre_point = pre_point2
        #print("in step", dist, angle_diff)
        # dist, angle_diff= self.normalize(d=dist, a=angle_diff)
        self.preview(action, preview=1)
        # print("____________________________preview______________________________")
        future_dist, future_angle_diff, _ = lateralenv.dist_diff(self, ep=0, limit_dist=1, limit_ang=0,
                                                                 stp=stp_cnt, pre_point=pre_point)
        # future_dist, future_angle_diff= self.normalize( d= future_dist, a=future_angle_diff)

        self.episode_length_cnt = self.episode_length_cnt - 1

        if dist > self.dist_limit or angle_diff > self.ang_limit1 or angle_diff < self.ang_limit2 or self.episode_length_cnt == 0:
            print("last dist", dist, "last angle", angle_diff)
            self.Done = 1
            # return self.vars_, self.state_,  bad_reward, 'nothing' , self.Done, pre_point
            return None, None, self.bad_reward, 'nothing', self.Done, None
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% calc reward %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # 3: based on car's vertical disance with the road
        else:
            dist, angle_diff = self.normalize(d=dist, a=angle_diff)
            future_dist, future_angle_diff = self.normalize(d=future_dist, a=future_angle_diff)
            weight = 1
            action_weight = -1
            preview_weight = 0.1
            # print("point", point, dist, "ang", angle_diff)
            # print("dist", dist, "ang", angle_diff)
            k1 = 1 / (dist ** 2 + angle_diff ** 2)
            k2 = 1 / (future_dist ** 2 + future_angle_diff ** 2)
            ep_len_weight = 1
            reward_calc = f'{weight} * {k1} + {preview_weight}*{k2} + {action_weight} * {action} + {ep_len_weight} * {ep_length}'
            # reward = - angle_diff

            ## 4: Sarah test
            # ------------------------
            reward = k1 * weight + k2 * preview_weight + action_weight * action + ep_len_weight * (
                        self.max_ep_length - self.episode_length_cnt)

            # print("dist", dist,"angd", angle_diff, "p dist",future_dist, "p angd", future_angle_diff)
            self.state_ = np.array([dist, angle_diff, future_dist, future_angle_diff])  # real state (not limited)
            # self.state_ = np.array([dist, angle_diff]) #real state (not limited)

            # for next step
            self.vars = self.vars_
            self.coordinates.append(self.vars[2:4, 0])
            self.vys.append(self.vars[0])

            return self.vars_, self.state_, reward, reward_calc, self.Done, pre_point  # state:(dist, ang_dif)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def render(self, ep, score, ep_length, pnt):
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(self.road_ep.coords.xy[0][0:50], self.road_ep.coords.xy[1][0:50], 'r')  # road
        if ep_length != 0:
            plt.plot(np.array(self.coordinates)[:, 0], np.array(self.coordinates)[:, 1], label=score)  # path
            # b=1
        #if(ep_length<10):
            # plt.xlim(pnt-100, pnt+50)
            # plt.ylim(-150, 150)
            # plt.gca().set_aspect('equal', adjustable='box')
            #plt.show()
        if ep % 10 == 0 and ep_length != 0:
            plt.legend()
            # plt.show()
            # plt.xlim(pnt, pnt+200)
            # plt.ylim(-150, 150)
            # plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(f"path{ep}.jpg")
            plt.cla()
            b = 0

    def reset(self, ep_pointer):  # before each episode
        self.Done = 0
        self.episode_length_cnt = self.max_ep_length
        self.coordinates = []
        self.nearestPiontCheck = []

        # a new section of the road excel is selected for each episode
        self.data_ep = self.road[ep_pointer:ep_pointer + int(self.max_ep_length / self.res), :]
        self.road_ep = geom.LineString(zip(self.x[ep_pointer: ep_pointer + int(self.max_ep_length / self.res)],
                                           self.y[ep_pointer: ep_pointer + int(self.max_ep_length / self.res)]))  # 500*2
        
        # the car starts on the road
        st_vy = 0;
        st_r = 0;
        st_x = self.data_ep[0, 0]  # + np.random.rand()
        # st_x = self.road[ep_pointer:ep_pointer+1,0]
        st_y = self.data_ep[0, 1]  # + np.random.rand()
        # st_y = self.road[ep_pointer+ep_pointer+1,1]
        # st_psi = self.heading_angle[ep_pointer]  # + np.random.rand()*0.01
        st_psi = (self.data_ep[1,1] - self.data_ep[0,1]) / (self.data_ep[1,0] - self.data_ep[0,0])

        st_pre_point = geom.Point(st_x, st_y)

        print("deeeeeeeeebuuuuuu", st_x, st_y,st_psi)

        self.vars = np.array([[st_vy, st_r, st_x, st_y, st_psi]], dtype='float64').T
        self.vars_tmp = np.array([[st_vy, st_r, st_x, st_y, st_psi]],
                                 dtype='float64').T  # is updated for both normal step and preview step

        # point0_ep = geom.Point(st_x, st_y)
        limited_dist0, limited_angle_diff0, pre_p = self.dist_diff(ep=0, limit_dist=1, limit_ang=0, stp=0,
                                                                   pre_point=st_pre_point)
        limited_dist0, limited_angle_diff0 = self.normalize(limited_dist0, limited_angle_diff0)

        self.preview(action=0, preview=1)
        future_limited_dist0, future_limited_ang0, _ = self.dist_diff(ep=0, limit_dist=1, limit_ang=0, stp=0,
                                                                      pre_point=pre_p)  # sefr
        future_limited_dist0, future_limited_ang0 = self.normalize(future_limited_dist0, future_limited_ang0)

        state0_ep = np.array([limited_dist0, limited_angle_diff0, future_limited_dist0, future_limited_ang0])  # (1,4)

        return state0_ep, st_pre_point