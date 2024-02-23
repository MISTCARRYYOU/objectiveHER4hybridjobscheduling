"""
Face to DRL-based CEDCS triggered simulation environment:
> The assignment of a computational task and a manufacturing task is regarded as a step.
> The time is a type of resources attributed to devices or servers.
"""
import random

from utilities import read_paras_txt, GoalAdjust
import copy
QN = 23
Q_upload = 100
Q_process = 2e-2
Q_logistics = 5e-1

reward_step_interval = 50  # how much step obtain the reward

HER_goal_update_freq = 5  # every 50 episodes update the goal
delta_gt = 0.05
import numpy as np
import math
import gym


class CEDCS_env(gym.Env):
    environment_name = 'CEDCS'
    id = 'CEDCS'
    action_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=float)
    reward_threshold = -10000
    trials = 100
    _max_episode_steps = 1e10

    tmp_look = []

    def __init__(self, file_path, CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum):
        # static:
        self.env_name = 'CEDCS_' + file_path.split('.t')[0][-10:]
        self.env_paras = read_paras_txt(file_path, CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum)  # scene paras
        print('\n', self.env_name, ' is now initialized !!!!!!!!!!!!!!!!!!!!!!')

        self.Cnum = Cnum
        self.M_OPTnum = M_OPTnum
        self.CE_Tnum = CE_Tnum
        self.Enum = Enum
        self.Dnum = Dnum
        self.M_Jnum = M_Jnum
        self.server_mode = None  # cloud or edge
        self.done = False
        self.reward_time = [stp for stp in range(reward_step_interval, self.CE_Tnum, reward_step_interval)]
        # print(self.reward_time, 6666666666666666666)
        # 计算每个边缘端到设备端的最短距离，计算一次即可
        self.nearest_device = [0 for _ in range(Enum)]
        for i in range(0, Enum, 1):
            min_dis = 0
            min_index = 0
            for j in range(0, Dnum, 1):
                if min_dis > self.env_paras.EtoD_Distance[i][j]:
                    min_dis = self.env_paras.EtoD_Distance[i][j]
                    min_index = j
            self.nearest_device[i] = min_index
        # 计算每个设备到边缘的最短距离，计算一次即可
        self.nearest_edge = [0 for _ in range(Dnum)]
        for i in range(0, Dnum, 1):
            min_dis = 1e10
            min_index = 0
            for j in range(0, Enum, 1):
                if min_dis > self.env_paras.EtoD_Distance[j][i]:
                    min_dis = self.env_paras.EtoD_Distance[j][i]
                    min_index = j
            self.nearest_edge[i] = min_index

        # got some statistical num
        comp_times = []
        comm_times = []
        for eve in self.env_paras.CETask_Property:
            comm_times.append(eve.Communication)
            comp_times.append(eve.Computation)
        self.comp_times_med = np.median(comp_times)
        self.comm_times_med = np.median(comm_times)

        self.dynamic_goal = None  # should be updated when initialized
        self.all_achieved_goals = []  # save all achieved goals

        self.episode_num = 0
        self.epi_rws = []  # save episode rewards

        self.goal_adjust = GoalAdjust()
        self.goal_obj_pairs = []  # save goal and obj pairs in each epi

        # dynamic:
        self.m_st = [[0.0 for _ in range(M_OPTnum)] for __ in range(M_Jnum)]  # start time of maf tasks
        self.m_et = [[0.0 for _ in range(M_OPTnum)] for __ in range(M_Jnum)]  # end time of maf tasks
        self.c_st = [0.0 for _ in range(CE_Tnum)]  # start time of cop tasks
        self.c_et = [0.0 for _ in range(CE_Tnum)]  # end time of cop tasks

        self.steps = 0  # num of steps in an episode

        self.cevar = [0 for _ in range(CE_Tnum)]  # which server for each cop task
        self.mvar = [[0 for _ in range(M_OPTnum)] for __ in range(M_Jnum)]  # which device for each maf subtask

        self.device_load = [[] for _ in range(Dnum)]  # working load of each device
        self.cetask_codevice = [[] for _ in range(CE_Tnum)]  # corresponding devices for each server whether in edge or cloud

        self.cloud_devices = [[] for _ in range(Cnum)]
        self.cloud_load = [[] for _ in range(Cnum)]

        self.edge_devices = [[] for _ in range(Enum)]
        self.edge_load = [[] for _ in range(Enum)]
        self.edge_device_comm = [{} for _ in range(Enum)]

        self.edge_smallest_rate = [1e10 for _ in range(Enum)]  # 每个边缘设备的最小率

        # each device connects the nearest edge for data forwarding.
        for i in range(Enum):
            # self.edge_device will not be empty for cloud trans
            self.edge_devices[i].append(self.nearest_device[i])

        self.comm_energys = []  # save step energy
        self.device_energys = []  # save step device energy
        self.step_makespan = []   # save step makespan

        # some variables for ablation study
        self.is_new1 = False
        self.is_new2 = False

    def reset(self, is_state_dict=False):
        self.is_state_dict = is_state_dict  # for HER DRL
        self.steps = 0
        self.m_st = [[0.0 for _ in range(self.M_OPTnum)] for __ in range(self.M_Jnum)]  # start time of maf tasks
        self.m_et = [[0.0 for _ in range(self.M_OPTnum)] for __ in range(self.M_Jnum)]  # end time of maf tasks
        self.c_st = [0.0 for _ in range(self.CE_Tnum)]  # start time of cop tasks
        self.c_et = [0.0 for _ in range(self.CE_Tnum)]  # end time of cop tasks

        self.latest_et = [0.0 for _ in range(self.CE_Tnum)]

        self.mt_logist = 0

        self.steps = 0  # num of steps in an episode

        self.cevar = [0 for _ in range(self.CE_Tnum)]  # which server for each cop task
        self.mvar = [[0 for _ in range(self.M_OPTnum)] for __ in range(self.M_Jnum)]  # which device for each maf subtask

        self.device_load = [[] for _ in range(self.Dnum)]  # working load of each device
        self.cetask_codevice = [[] for _ in range(self.CE_Tnum)]  # corresponding devices for each server whether in edge or cloud

        self.cloud_devices = [[] for _ in range(self.Cnum)]
        self.cloud_load = [[] for _ in range(self.Cnum)]

        self.edge_devices = [[] for _ in range(self.Enum)]
        self.edge_load = [[] for _ in range(self.Enum)]
        self.edge_device_comm = [{} for _ in range(self.Enum)]

        self.edge_smallest_rate = [1e10 for _ in range(self.Enum)]  # 每个边缘设备的最小率

        # each device connects the nearest edge for data forwarding.
        for i in range(self.Enum):
            # self.edge_device will not be empty for cloud trans
            self.edge_devices[i].append(self.nearest_device[i])

        self.comm_energys = []  # save step energy
        self.device_energys = []
        self.step_makespan = []  # save stemp makespan
        self.actions = []
        self.epi_bias = []
        self.done = False

        self.epi_rewards = []

        self.epi_max_ave_time = 0

        self.last_reward = 0

        self.server_modes = []

        # self.episode_num += 1

        if self.is_state_dict:
            if (self.dynamic_goal is None) or (self.is_new2 is True):  # the first time to run this class
                # print(self.all_achieved_goals)
                if self.is_new2 is False:
                    assert len(self.all_achieved_goals) == 0
                self.dynamic_goal = self.randomly_set_the_init_goal()
                self.all_achieved_goals.append(self.dynamic_goal)
            else:  # use the best goal
                if self.episode_num % HER_goal_update_freq == 0 and self.episode_num != 0:
                    # self.dynamic_goal = min(self.all_achieved_goals)
                    # print('\n', sum(self.epi_rws[-HER_goal_update_freq:]) / HER_goal_update_freq, self.dynamic_goal)

                    self.goal_adjust.train_goal_net()
                    x = []
                    for pari_i in range(-5, 0, 1):
                        x += [self.goal_obj_pairs[pari_i][1]]
                    self.dynamic_goal = self.goal_adjust.predict_delta(x) - 0.05
                    # self.dynamic_goal -= (sum(self.epi_rws[-HER_goal_update_freq:]) / HER_goal_update_freq) * delta_gt

                    # if self.CE_Tnum == 100:
                    #     self.dynamic_goal = 2060.7 / 1e4 + 0.05
                    # elif self.CE_Tnum == 200:
                    #     self.dynamic_goal = 5420.2 / 1e4 + 0.1
                    # elif self.CE_Tnum == 300:
                    #     self.dynamic_goal = 7806.38 / 1e4 + 0.2
                    # elif self.CE_Tnum == 400:
                    #     self.dynamic_goal = 10266.96 / 1e4 + 0.3
                    # else:
                    #     assert False
        return self.get_state()

    # action is seven dims \in (0, 1)
    def step(self, action, is_state_dict=False):
        self.is_state_dict = is_state_dict  # for HER DRL
        if type(action) is not list:
            action = action.tolist()
        # print(action)
        # assign index
        i_task = self.steps
        # cop tasks
        if action[0] < 0.5:
            sorted_avail_edges = self.sort_by_load('edge', self.env_paras.CETask_Property[i_task].AvailEdgeServerList)
            tmp_index = int((1 - 2*action[0]) * (len(sorted_avail_edges) - 1))

            # For ablation study in major revise------------------------
            if self.is_new1 is True:
                self.cevar[i_task] = self.env_paras.CETask_Property[i_task].AvailEdgeServerList[tmp_index]
            else:
                self.cevar[i_task] = sorted_avail_edges[tmp_index]
            # ----------------------------------------------------------

            self.server_mode = 'edge'
        else:
            sorted_avail_clouds = self.sort_by_load('cloud', [each_cloud for each_cloud in range(self.Cnum)])
            tmp_index = int(2*(1 - action[0]) * (len(sorted_avail_clouds)-1))

            # For ablation study in major revise-----------------------
            if self.is_new1 is True:
                self.cevar[i_task] = [each_cloud for each_cloud in range(self.Cnum)][tmp_index]
            else:
                self.cevar[i_task] = sorted_avail_clouds[tmp_index]
            # ----------------------------------------------------------

            self.server_mode = 'cloud'
        self.server_modes.append(self.server_mode)
        # maf tasks
        last_device = None
        for op_j in range(self.M_OPTnum):
            tmp_avail_devices = self.env_paras.AvailDeviceList[i_task * self.M_OPTnum + op_j]
            if op_j == 0:  # the fist do not need the distance, idle ...
                sorted_avail_devices = self.sort_by_load('device', tmp_avail_devices)
                self.mvar[i_task][op_j] = sorted_avail_devices[0]
            else:
                # more near 0.5, the scope should be larger
                if action[1] < 0.5:  # distance first
                    if self.is_new1 is True:
                        chosen = tmp_avail_devices[int(1 - 2*action[1]) * len(tmp_avail_devices)]
                    else:
                        sorted_avail_devices_by_distance = self.sort_by_dist(last_device, tmp_avail_devices)
                        scope = sorted_avail_devices_by_distance[:1+int(2*action[1]*(len(sorted_avail_devices_by_distance)-2))]
                        chosen = self.sort_by_load('device', scope)[0]
                else:  # load first
                    if self.is_new1 is True:
                        chosen = tmp_avail_devices[int(2 - 2*action[1]) * len(tmp_avail_devices)]
                    else:
                        sorted_avail_devices = self.sort_by_load('device', tmp_avail_devices)
                        scope = sorted_avail_devices[:1+int(2*(1-action[1])*(len(sorted_avail_devices)-2))]
                        chosen = self.sort_by_dist(last_device, scope)[0]
                self.mvar[i_task][op_j] = chosen
            last_device = self.mvar[i_task][op_j]

            # self.mvar[i_task][op_j] = tmp_avail_devices[int(action[0+op_j] * (len(tmp_avail_devices) - 1))]

        # now self.cevar and self.mvar are full of servers and devices indexed by i_task
        # then update the env global state based on current chosen and previous settlements
        self.update_time_and_energy_incementally(i_task)
        self.actions.append(action)

        if self.CE_Tnum == self.steps + 1:  # the last task has been allocated
            self.done = True
            self.episode_num += 1
            # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            # self.print_selection()
        else:
            self.done = False
            self.steps += 1
        self.step_makespan.append(self.calculate_makespan())  # save step makespan
        state = self.get_state()
        if is_state_dict:
            reward = self.get_reward_HER()
        else:
            reward = self.get_reward()
        self.epi_rewards.append(reward)
        # print(self.steps, reward, self.calculate_makespan(), self.comm_energys[-1], self.server_mode)

        if self.done:  # record
            actions_count = np.array(self.actions)
            bias_count = np.array(self.epi_bias)
            if is_state_dict:
                with open('./my_data_and_graph/' + self.env_name + 'logs.txt', 'a') as f:
                    print(round(self.obtain_obj(), 2), sum(self.epi_rewards), ' || ', round(self.calculate_makespan(), 2),
                          round(self.calculate_total_energy(), 2)
                          , ' || ',
                          self.goal_obj_pairs[-1]
                          , file=f)
            else:
                with open('./my_data_and_graph/' + self.env_name + 'logs.txt', 'a') as f:
                    print(round(self.obtain_obj(), 2), sum(self.epi_rewards), ' || ', round(self.calculate_makespan(), 2),
                          round(self.calculate_total_energy(), 2)
                          , file=f)
        return state, reward, self.done, {}

    def get_state(self):  # This will be pretty important
        # only percepts the device and server that are available to the next one ? seems to break the Markov chain
        # statistical features of resources of the global
        state = [self.steps / self.CE_Tnum]
        # state = []
        bias = [#self.compute_bias4m_et(self.m_et)/(self.steps*1e1 + 1e1),
                #self.compute_bias4c_et(self.c_et)/(self.steps*1e1 + 1e1),
                #self.compute_load_bias(self.edge_load)/1e1,
                self.compute_load_bias(self.cloud_load)/1e1,
                #self.compute_load_bias(self.device_load)/1e2
                ]
        state += bias
        self.epi_bias.append(bias)
        # current situation of next available ones
        state += [self.env_paras.CETask_Property[self.steps].Computation / 1e2,
                  self.env_paras.CETask_Property[self.steps].Communication / 1e3,
                  ]
        edge_loads = []
        for eve in self.env_paras.CETask_Property[self.steps].AvailEdgeServerList:
            edge_loads.append(self.edge_load[eve])
        state += [self.compute_load_bias(edge_loads)]
        for opj in range(self.M_OPTnum):
            if opj != 0:
                tmp_avail_devices = self.env_paras.AvailDeviceList[self.steps * self.M_OPTnum + opj]
                loads = [self.device_load[eve] for eve in tmp_avail_devices]
                state += [self.compute_load_bias(loads)]
        max_s = max(state)
        if max_s == 0: max_s = 1.0
        for i in range(len(state)):
            if i != 0:  # the firt dont change
                state[i] = state[i] / max_s

        if not self.is_state_dict:
            return np.array(state)
        else:
            return {
                'observation': np.array(state),
                'desired_goal': np.array([self.dynamic_goal]),
                'achieved_goal': np.array([self.obtain_obj() / 1e4])
            }

    def get_reward(self):
        if self.steps in self.reward_time:
            current_reward = -0.001*self.obtain_obj()
            r = current_reward - self.last_reward
            self.last_reward = current_reward
            return r
        elif self.done:
            return -0.001 * self.obtain_obj()
        else:
            return 0

    def get_reward_HER(self):
        if self.done:
            epix_achieved = self.obtain_obj() / 1e4
            # print(epix_achieved, self.dynamic_goal)
            self.goal_obj_pairs.append([self.dynamic_goal, epix_achieved])

            if epix_achieved <= self.dynamic_goal:
                self.all_achieved_goals.append(epix_achieved)
                rt = 1.0
            else:
                rt = -1.0
            self.epi_rws.append(rt)
            return rt
        else:
            return 0.0

    def obtain_obj(self):
        return 0.5 * self.calculate_makespan() + 0.5 * self.calculate_total_energy()

    # used in the final step
    def calculate_total_energy(self):
        energy = sum(self.comm_energys) + sum(self.device_energys)
        for i in range(0, self.Cnum, 1):
            if len(self.cloud_load[i]) == 0:
                continue
            u_ratio = int(len(self.cloud_load[i]) / 20.0 * 10)
            if u_ratio > 10:
                u_ratio = 10
            time_expand = 0
            for star_iter in self.cloud_load[i]:
                if self.c_et[star_iter] - self.c_st[star_iter] > time_expand:
                    assert self.c_st[star_iter] == 0.
                    time_expand = self.c_et[star_iter] - self.c_st[star_iter]
            energy += self.env_paras.EnergyList[u_ratio] * time_expand / 1000.0  # 云服务器开着就费电， energy list存的就是多少比率占用多少的意思

        for i in range(0, self.Enum, 1):
            if len(self.edge_load[i]) == 0:
                continue
            u_ratio = int(len(self.edge_load[i]) / 6.0 * 10)
            if u_ratio > 10:
                u_ratio = 10
            time_expand = 0
            for star_iter in self.edge_load[i]:
                if self.c_et[star_iter] - self.c_st[star_iter] > time_expand:
                    time_expand = self.c_et[star_iter] - self.c_st[star_iter]
            energy += self.env_paras.EnergyList[u_ratio] * time_expand / 1000.0  # 边缘服务器开着就费电
        # print(sum(self.comm_energys), sum(self.device_energys), 6666666666)
        return energy

    def calculate_makespan(self):
        return max(max(self.c_et), max(max(self.m_et)))

    # measure the bias of alist
    # when there is less task, the influence of bias should be less
    def compute_load_bias(self, alist):
        if type(alist[0]) == list:  # 2-d list
            res = [len(eve) for eve in alist]
        else:
            res = [eve for eve in alist]
        ave = int(sum(res) / len(res))  # == 1 mostly
        # print(sum(res))
        bias = 0
        for eve in res:
            if eve - ave >= 0:  # exceed the load
                bias += (eve - ave)
        return bias

    # specially for self.m_et
    def compute_bias4m_et(self, alist):
        res = []
        for eve in alist:
            if eve[-1] != 0.0:
                res.append(eve[-1])
        if len(res) == 0:
            return 0
        ave = sum(res) / len(res)
        bias = 0
        for eve in res:
            bias += abs(eve - ave)
        return bias

    def compute_bias4c_et(self, alist):
        res = []
        for eve in alist:
            if eve != 0.0:
                res.append(eve)
        if len(res) == 0:
            return 0
        ave = sum(res) / len(res)
        bias = 0
        for eve in res:
            bias += abs(eve - ave)
        return bias

    # update the time and energy of i_task based on previous accumulation
    def update_time_and_energy_incementally(self, i_task):
        mt_energy_consume = 0
        #  update the device load
        for op_j in range(self.M_OPTnum):
            tmp_target_device = self.mvar[i_task][op_j]  # device to be assigned
            if op_j == 0:  # The first operation in the task
                if len(self.device_load[tmp_target_device]) == 0:  # The device is fresh new
                    self.m_st[i_task][op_j] = 0
                    self.m_et[i_task][op_j] = self.env_paras.MTask_Time[self.M_OPTnum*i_task + op_j]
                    mt_energy_consume += self.env_paras.MTask_Time[self.M_OPTnum*i_task + op_j] * Q_process
                else:  # The device has been occupied
                    last_op = self.device_load[tmp_target_device][-1]
                    self.m_st[i_task][op_j] = self.m_et[int(last_op / self.M_OPTnum)][last_op % self.M_OPTnum]
                    self.m_et[i_task][op_j] = self.m_st[i_task][op_j] + self.env_paras.MTask_Time[self.M_OPTnum*i_task+op_j]
                    mt_energy_consume += self.env_paras.MTask_Time[self.M_OPTnum*i_task+op_j] * Q_process
            else:  # not the first operation in the task
                if len(self.device_load[tmp_target_device]) == 0:
                    self.m_st[i_task][op_j] = self.m_et[i_task][op_j-1]
                    self.m_et[i_task][op_j] = self.m_st[i_task][op_j] + \
                                              self.env_paras.MTask_Time[self.M_OPTnum*i_task + op_j] + \
                                              self.env_paras.DtoD_Distance[self.mvar[i_task][op_j-1]][tmp_target_device] / (100000.0 / 3600.0)
                    mt_energy_consume += self.env_paras.MTask_Time[self.M_OPTnum*i_task + op_j] * Q_process + Q_logistics*(self.env_paras.DtoD_Distance[self.mvar[i_task][op_j-1]][tmp_target_device] / (100000.0 / 3600.0))
                    self.mt_logist += Q_logistics*(self.env_paras.DtoD_Distance[self.mvar[i_task][op_j-1]][tmp_target_device] / (100000.0 / 3600.0))
                else:
                    last_op = self.device_load[tmp_target_device][-1]
                    self.m_st[i_task][op_j] = max(self.m_et[i_task][op_j-1], self.m_et[int(last_op / self.M_OPTnum)][last_op % self.M_OPTnum])
                    self.m_et[i_task][op_j] = self.m_st[i_task][op_j] + \
                                              self.env_paras.MTask_Time[self.M_OPTnum*i_task + op_j] + \
                                              self.env_paras.DtoD_Distance[self.mvar[i_task][op_j-1]][tmp_target_device] / (100000.0 / 3600.0)
                    mt_energy_consume += self.env_paras.MTask_Time[self.M_OPTnum*i_task + op_j] * Q_process + \
                                         (self.env_paras.DtoD_Distance[self.mvar[i_task][op_j-1]][tmp_target_device] / (100000.0 / 3600.0)) * Q_logistics
                    self.mt_logist += (self.env_paras.DtoD_Distance[self.mvar[i_task][op_j-1]][tmp_target_device] / (100000.0 / 3600.0)) * Q_logistics

            # update the device load of the target device
            self.device_load[tmp_target_device].append(self.M_OPTnum * i_task + op_j)
            # update the devices connected to the server
            if self.mvar[i_task][op_j] not in self.cetask_codevice[i_task]:  # if the connection was not built
                self.cetask_codevice[i_task].append(self.mvar[i_task][op_j])
            if self.server_mode == 'cloud':
                if self.mvar[i_task][op_j] not in self.cloud_devices[self.cevar[i_task]]:
                    self.cloud_devices[self.cevar[i_task]].append(self.mvar[i_task][op_j])
            else:
                if self.mvar[i_task][op_j] not in self.edge_devices[self.cevar[i_task]]:
                    self.edge_devices[self.cevar[i_task]].append(self.mvar[i_task][op_j])

        # update servers
        if self.server_mode == 'cloud':
            self.cloud_load[self.cevar[i_task]].append(i_task)
        else:
            self.edge_load[self.cevar[i_task]].append(i_task)

        # now all servers and devices are updated with current chosen ones
        # then update the edge devices; only the selected one will be updated !!
        if self.server_mode == 'edge':
            bottom_sum = 0.0
            for iter in self.edge_devices[self.cevar[i_task]]:
                bottom_sum += QN / pow(self.env_paras.EtoD_Distance[self.cevar[i_task]][iter] / 1000, 1.0)

            for iter in self.edge_devices[self.cevar[i_task]]:
                current_gain = QN / pow(self.env_paras.EtoD_Distance[self.cevar[i_task]][iter] / 1000, 1.0)
                transmission_rate = 20 * math.log2(
                    1 + current_gain / abs(bottom_sum - current_gain - 100)) / 8.0  # Mbps -> MB / s
                self.edge_device_comm[self.cevar[i_task]][iter] = transmission_rate
                if transmission_rate < self.edge_smallest_rate[self.cevar[i_task]]:  # 大概率是要小的
                    self.edge_smallest_rate[self.cevar[i_task]] = transmission_rate
                # self.tmp_look.append(transmission_rate)
                # print(self.tmp_look)

        # then compute the time and energy of cop task indexed by i_task
        t_comm = 0
        energy_comm = 0
        tmp_target_server = self.cevar[i_task]
        # communication time
        if self.server_mode == 'cloud':
            for iter in self.cetask_codevice[i_task]:
                # cur_comm = self.env_paras.CETask_Property[i_task].Communication * 10 / (1000 * self.edge_smallest_rate[self.nearest_edge[iter]])
                cur_comm = self.env_paras.CETask_Property[i_task].Communication * 10 / Q_upload

                energy_comm += cur_comm * QN / 1000
                t_comm = max(t_comm, cur_comm)
            # print('cloud ene comm ', energy_comm)
        else:  # edge mode
            for iter in self.cetask_codevice[i_task]:
                cur_comm = self.env_paras.CETask_Property[i_task].Communication * 10 / (1000 * self.edge_device_comm[tmp_target_server][iter])
                energy_comm += cur_comm * QN / 1000
                t_comm = max(t_comm, cur_comm)
            # print('edge ene comm ', energy_comm)
        # computational time
        if self.server_mode == 'cloud':
            t_comp = self.env_paras.CETask_Property[tmp_target_server].Computation / 4.1
        else:
            if len(self.edge_load[tmp_target_server]) < 6:
                t_comp = self.env_paras.CETask_Property[tmp_target_server].Computation / 2.2
            else:
                t_comp = self.env_paras.CETask_Property[tmp_target_server].Computation / 2.2 * len(self.edge_load[tmp_target_server])
        # update the real time  the four constraints
        # if self.env_paras.CETask_Property[i_task].Job_Constraints == 0 or self.env_paras.CETask_Property[i_task].Job_Constraints == 2:
        #     self.c_st[i_task] = min(self.c_st[i_task], self.m_st[i_task][0])
        # elif self.env_paras.CETask_Property[i_task].Job_Constraints == 1 or self.env_paras.CETask_Property[i_task].Job_Constraints == 3:
        #     self.c_st[i_task] = max(self.c_st[i_task], self.m_st[i_task][0])
        self.c_et[i_task] = self.c_st[i_task] + t_comp + t_comm
        # if self.env_paras.CETask_Property[i_task].Job_Constraints == 1 or self.env_paras.CETask_Property[i_task].Job_Constraints == 2:
        #     self.c_et[i_task] = min(self.c_et[i_task], self.m_et[i_task][-1])
        # elif self.env_paras.CETask_Property[i_task].Job_Constraints == 0 or self.env_paras.CETask_Property[i_task].Job_Constraints == 3:
        #     self.c_et[i_task] = max(self.c_et[i_task], self.m_et[i_task][-1])
        # now the time for each cop task is updated successfully
        # the energy consumption can not be calculated incrementally, as different allocations influence the power
        # only the communication power can be obtained
        self.comm_energys.append(energy_comm)
        self.device_energys.append(mt_energy_consume)
        # update the latest end time
        self.latest_et[i_task] = max(self.c_et[i_task], self.m_et[i_task][-1])

    # sort the load in three modes: cloud, edge, device
    def sort_by_load(self, mode, alist):
        res = []
        temp = []
        if mode == 'cloud':
            for eve in alist:
                temp.append(len(self.cloud_load[eve]))
        elif mode == 'edge':
            for eve in alist:
                temp.append(len(self.edge_load[eve]))
        elif mode == 'device':
            for eve in alist:
                temp.append(len(self.device_load[eve]))
        else:
            assert False
        # from small to big
        seq = np.argsort(temp)
        for eve2 in seq:
            res.append(alist[eve2])
        return res

    # sort the distance
    def sort_by_dist(self, cur_device, next_device_list):
        temp = []
        res = []
        for eve in next_device_list:
            temp.append(self.env_paras.DtoD_Distance[cur_device][eve])
        seq = np.argsort(temp)
        for eve2 in seq:
            res.append(next_device_list[eve2])
        return res

    def compute_reward(self, next_goal, new_goal, dontknow):
        if next_goal <= new_goal:
            return 1.0
        else:
            return -1.0

    # obtain the initial goal randomly
    def randomly_set_the_init_goal(self):
        if self.CE_Tnum == 100:
            return 2060.7 / 1e4
        elif self.CE_Tnum == 200:
            return 5420.2 / 1e4
        elif self.CE_Tnum == 300:
            return 7806.38 / 1e4
        elif self.CE_Tnum == 400:
            return 10266.96 / 1e4
        elif self.CE_Tnum == 500:
            return 12114.72 / 1e4
        else:
            assert False

