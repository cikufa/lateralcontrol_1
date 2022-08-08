import numpy as np
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

        self.Done = 0
        self.coordinates = []
        self.nearestPiontCheck = []

        self.vars = np.zeros((5, 1))
        self.vars_ = np.zeros((5, 1), dtype='float64')  # is only updated for normal step
        self.vars_tmp = np.zeros((5, 1))  # is updated for both normal step and preview step

    def dist_diff(self, ep, limit_dist, limit_ang, stp, pre_point):  # =geom.Point(0,0)):
        vy, r, x, y, psi = self.vars_tmp

        # 3: based on car's vertical disance with the road
        minus = self.data_ep - np.array((x, y)).reshape([1,2])
        distarr = np.sqrt(minus[:, 0] ** 2 + minus[:, 1] ** 2)
        ind = np.argmin(distarr)

        point = geom.Point(x, y)
        dist = point.distance(self.road_ep)
        limited_dist = max(dist, 0.01)  # for makhraj problems

        road_slope_rad = np.arctan2((self.data_ep[ind + 1, 1] - self.data_ep[ind, 1]),(self.data_ep[ind + 1, 0] - self.data_ep[ind, 0])) if ind != len(self.data_ep) - 1 else np.arctan2((self.data_ep[ind, 1] - self.data_ep[ind - 1, 1]),(self.data_ep[ind, 0] - self.data_ep[ind - 1, 0]))
        angle_diff = abs(road_slope_rad - psi)[0]
        limited_angle_diff = max(angle_diff, 0.005)

        nearestP = nearest_points(self.road_ep, point)[0]

        if limit_dist == 1:
            if limit_ang == 1:
                return limited_dist, limited_angle_diff, nearestP
            elif limit_ang == 0:
                return limited_dist, angle_diff, nearestP
        else:
            return dist, angle_diff, nearestP

    def reset(self, ep_pointer):  # before each episode
        if ep_pointer > (self.road.shape[0] - 300):  # =max_episode_length. resets the road and stars from the begining
            ep_pointer = 0

        self.Done = 0
        self.coordinates = []
        self.nearestPiontCheck = []

        # a new section of the road excel is selected for each episode
        self.data_ep = self.road[ep_pointer:ep_pointer + int(self.max_ep_length / self.res), :]


        # the car starts on the road
        st_vy = 0;
        st_r = 0;
        st_x = self.data_ep[0, 0]  # + np.random.rand()
        # st_x = self.road[ep_pointer:ep_pointer+1,0]
        st_y = self.data_ep[0, 1]  # + np.random.rand()
        # st_y = self.road[ep_pointer+ep_pointer+1,1]
        # st_psi = self.heading_angle[ep_pointer]  # + np.random.rand()*0.01
        st_psi = np.arctan2((self.data_ep[1, 1] - self.data_ep[0, 1]), (self.data_ep[1, 0] - self.data_ep[0, 0]))

        st_pre_point = geom.Point(st_x, st_y)
        self.coordinates.append([st_x, st_y])


        self.vars = np.array([[st_vy, st_r, st_x, st_y, st_psi]], dtype='float64').T
        self.vars_tmp = np.array([[st_vy, st_r, st_x, st_y, st_psi]],
                                 dtype='float64').T  # is updated for both normal step and preview step

        # point0_ep = geom.Point(st_x, st_y)
        limited_dist0, limited_angle_diff0, pre_p = self.dist_diff(ep=0, limit_dist=1, limit_ang=0, stp=0,
                                                                   pre_point=st_pre_point)
        limited_dist0, limited_angle_diff0 = self.normalize(limited_dist0, limited_angle_diff0)

        self.preview(0, t, preview=1)
        future_limited_dist0, future_limited_ang0, _ = self.dist_diff(ep=0, limit_dist=1, limit_ang=0, stp=0,
                                                                      pre_point=pre_p)  # sefr
        future_limited_dist0, future_limited_ang0 = self.normalize(future_limited_dist0, future_limited_ang0)

        state0_ep = np.array([limited_dist0, limited_angle_diff0, future_limited_dist0, future_limited_ang0])  # (1,4)

        return state0_ep, ep_pointer