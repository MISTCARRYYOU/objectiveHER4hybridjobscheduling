import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
import timeit
from torch.utils.data import Dataset
from collections import deque


# 云边任务结构体
class CETask:
    def __init__(self):
        self.Computation = 0.
        self.Communication = 0.
        self.Precedence = []
        self.Interact = []
        self.Start_Pre = []
        self.End_Pre = []
        self.Job_Constraints = 0
        self.AvailEdgeServerList = []

class GoalDataset(Dataset):
    def __init__(self, data):
        self.y_data = data.T[-1]
        self.x_data = data.T[:-1].T
        self.dataset_len = data.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.dataset_len


class GoalNet(nn.Module):
    def __init__(self):
        super(GoalNet, self).__init__()
        self.layer1 = nn.Linear(5, 32)
        self.layer2 = nn.Linear(32, 1)

    def forward(self, input):
        x = F.relu(self.layer1(input))
        x = abs(self.layer2(x))
        return x


class GoalAdjust:
    def __init__(self):
        print('GoalAdjust is initialized .....................')
        self.my_model = GoalNet()
        self.opt = torch.optim.Adam(self.my_model.parameters(), lr=0.001)
        self.loss_func = torch.nn.MSELoss()
        self.buffer = deque(maxlen=201)

    # x is a 10-dim list [g, j,g, j,g, j,g, j,g, j]
    # every five steps predict once and keep this choice for the next five steps
    def predict_delta(self, x):  # need padding
        with torch.no_grad():
            y = self.my_model(torch.tensor(x).view(1, -1))
            self.buffer.append(x + [-1])
            # print(x, y)
            if abs(y - sum(x)/len(x)) > 0.3:
                return sum(x)/len(x) - 0.1
            else:
                if y > sum(x)/len(x):
                    return sum(x)/len(x) - 0.1
                else:
                    return y
            # print(y)
            # if y[0][0] > y[0][1]:
            #     self.buffer.append(x + [0])
            #     return 3e-4
            # else:
            #     # print('Iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii',y)
            #     self.buffer.append(x + [1])
            #     return 2e-5

    def relabel_buffer(self):
        assert len(self.buffer) > 2
        for i in range(len(self.buffer)-1):
            next_samples = self.buffer[i+1]
            self.buffer[i][-1] = sum(next_samples[:-1]) / len(next_samples[:-1])
        return list(self.buffer)[:-1]

    # data: [(g, j, g,j,g,j,g,j,g,j,y),...]
    def train_goal_net(self):
        if len(self.buffer) > 20:  # 250 episodes after
            losses = []
            labeled_data = self.relabel_buffer()
            train_data = torch.tensor(labeled_data)
            current_dataset = GoalDataset(train_data)
            current_dataloader = DataLoader(dataset=current_dataset, batch_size=4)
            for EPO in range(10):
                for i, dt in enumerate(current_dataloader):
                    x, y = dt
                    y_hat = self.my_model(x)
                    loss = self.loss_func(y_hat, y[:, None])
                    self.opt.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    self.opt.step()
            with torch.no_grad():
                with open('./my_data_and_graph/losses/dnnloss.txt', 'a') as f:
                    print(len(self.buffer), sum(losses)/len(losses), file=f)


# 生成二维行列矩阵
def CreateMatrix(row, col):
    return np.zeros([row, col]).tolist()


def file2stream(path):
    res = []
    with open(path, 'r') as f:
        for eve in f:
            res += [float(each) if '.' in each else int(each) for each in eve.strip('\n').split()]
    return res


# 计算两个个体间的欧式距离
def OE_distance(pop1, pop2):
    p1 = np.array(pop1)
    p2 = np.array(pop2)
    return np.sqrt(np.sum((p1 -p2)**2))


# 将十进制转化为二进制，pop size
def onehot_coding(num, pop_size):
    bits = len(str(bin(pop_size))) - 2
    tmp = str(bin(num))
    addedzero = bits + 2 - len(tmp)
    res = '0'*addedzero + tmp[2:]
    res2 = []
    for eve in res:
        res2.append(int(eve))
    return res2

# for i in range(30):
#     print(onehot_coding(i, 30))


class Forread:
    def __init__(self, CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum):
        self.CETask_Property = [CETask() for _ in range(CE_Tnum)]  # 传入的参数不知道做什么？
        self.MTask_Time = [.0 for _ in range(M_Jnum * M_OPTnum)]
        self.EtoD_Distance = CreateMatrix(Enum, Dnum)
        self.DtoD_Distance = CreateMatrix(Dnum, Dnum)
        self.AvailDeviceList = [[] for _ in range(M_Jnum * M_OPTnum)]
        self.EnergyList = [0. for _ in range(11)]

        # self.CloudDevices = [[] for _ in range(Cnum)]
        # self.EdgeDevices = [[] for _ in range(Enum)]
        # self.CloudLoad = [[] for _ in range(Cnum)]
        # self.EdgeLoad = [[] for _ in range(Enum)]
        # self.DeviceLoad = [[] for _ in range(Dnum)]
        # self.CETask_coDevice = [[] for _ in range(CE_Tnum)]
        # self.Edge_Device_comm = [{} for _ in range(Enum)]
        # self.ST = CreateMatrix(M_Jnum, M_OPTnum)
        # self.ET = CreateMatrix(M_Jnum, M_OPTnum)
        # self.CE_ST = [0. for _ in range(CE_Tnum)]
        # self.CE_ET = [0. for _ in range(CE_Tnum)]


# 读取参数文件
def read_paras_txt(road_path, CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum):

    self = Forread(CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum)
    fs = file2stream(road_path)
    for i in range(0, Enum, 1):
        for j in range(0, Dnum, 1):
            self.EtoD_Distance[i][j] = fs.pop(0)

    for i in range(0, Dnum, 1):
        for j in range(0, Dnum, 1):
            self.DtoD_Distance[i][j] = fs.pop(0)

    for i in range(0, M_Jnum * M_OPTnum, 1):
        self.MTask_Time[i] = fs.pop(0)

    vec_num = 0
    for i in range(0, CE_Tnum, 1):

        self.CETask_Property[i].Computation = fs.pop(0)
        self.CETask_Property[i].Communication = fs.pop(0)

        vec_num = fs.pop(0)
        self.CETask_Property[i].Precedence = []
        for j in range(0, vec_num, 1):
            value = fs.pop(0)
            self.CETask_Property[i].Precedence.append(value)

        vec_num = fs.pop(0)
        self.CETask_Property[i].Interact = []
        for j in range(0, vec_num, 1):
            value = fs.pop(0)
            self.CETask_Property[i].Interact.append(value)

        vec_num = fs.pop(0)
        self.CETask_Property[i].Start_Pre = []
        for j in range(0, vec_num, 1):
            value = fs.pop(0)
            self.CETask_Property[i].Start_Pre.append(value)

        vec_num = fs.pop(0)
        self.CETask_Property[i].End_Pre = []
        for j in range(0, vec_num, 1):
            value = fs.pop(0)
            self.CETask_Property[i].End_Pre.append(value)
        self.CETask_Property[i].Job_Constraints = fs.pop(0)

    for i in range(0, M_Jnum, 1):
        for j in range(0, M_OPTnum, 1):
            vec_num = fs.pop(0)
            self.AvailDeviceList[i * M_OPTnum + j] = []
            for k in range(0, vec_num, 1):
                value = fs.pop(0)
                self.AvailDeviceList[i * M_OPTnum + j].append(value)

    for i in range(0, CE_Tnum, 1):
        vec_num = fs.pop(0)
        self.CETask_Property[i].AvailEdgeServerList = []
        for j in range(0, vec_num, 1):
            value = fs.pop(0)
            self.CETask_Property[i].AvailEdgeServerList.append(value)

    for i in range(0, 11, 1):
        self.EnergyList[i] = fs.pop(0)

    assert len(fs) == 0  # 保证文件都读完了，读取文件与C++是一致的
    return self


# record the env variables
def record_env_variables(environment_class):
    res = []
    res.append(environment_class.mt_logist)
    res.append(sum(environment_class.comm_energys))
    res.append(sum(environment_class.device_energys))

    res.append(environment_class.compute_load_bias(environment_class.edge_load) / 1e2)
    res.append(environment_class.compute_load_bias(environment_class.cloud_load) / 1)
    res.append(environment_class.compute_load_bias(environment_class.device_load) / 1e2)

    res2 = ''
    for eve in res:
        res2 += str(round(eve, 2))
        res2 += ' '
    return res2

def get_txt_res(txtpath):
    with open(txtpath, 'r') as f:
        cont = [eve for eve in f]
        res = {}
        tmp_name = None
        for eve in cont:
            if "--" in eve:
                tmp_name = eve.split(' :')[0]
                res[tmp_name] = []
            else:
                res[tmp_name].append([float(eve.split()[0]), float(eve.split()[1])])
        return res


def draw_training_result(root=''):
    files = [
        './my_data_and_graph/CEDCS__200_seed1logs.txt',
        './my_data_and_graph/CEDCS__300_seed1logs.txt',
        './my_data_and_graph/CEDCS__400_seed1logs.txt',
        './my_data_and_graph/CEDCS__500_seed1logs.txt',
    ]

    env_index = [200,300,400,500]

    for ind, each_file in enumerate(files):
        if os.path.exists(each_file) is False:
            print(each_file + ' did not exist !!!')
            continue
        res_dict = get_txt_res(each_file)
        # print(res_dict)
        obj = []
        rwd = []
        labels = []
        for key in res_dict.keys():
            tmp1 = []
            tmp2 = []
            for eve in res_dict[key]:
                tmp1.append(eve[0])
                tmp2.append(eve[1])
            obj.append(tmp1)
            rwd.append(tmp2)
            labels.append(key)
        plt.figure(ind)
        for i in range(len(labels)):
            plt.plot(obj[i], label=labels[i])
        plt.legend()
        plt.title('Objective')
        plt.savefig('./my_data_and_graph/look/obj_' + root + str(env_index[ind]) + '.png')
        plt.figure(ind+5)
        for i in range(len(labels)):
            plt.plot(rwd[i], label=labels[i])
        plt.legend()
        plt.title('Reward')
        plt.savefig('./my_data_and_graph/look/rwd_' + root + str(env_index[ind]) + '.png')


if __name__ == '__main__':
    draw_training_result('look-')
