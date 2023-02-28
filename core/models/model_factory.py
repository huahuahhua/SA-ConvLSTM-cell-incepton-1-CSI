import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import SA_ConvLSTM
# from core.models import STAU
# from core.models import AAU
# from core.models import STAUv2
# from core.models import AAUv2
import torch.optim.lr_scheduler as lr_scheduler


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_channel = configs.img_channel * (configs.patch_size ** 2)
        self.num_layers = configs.num_layers
        networks_map = {
            'ConvLSTM': SA_ConvLSTM.RNN,
            # 'stau': STAU.RNN,
            # 'aau': AAU.RNN,
            # 'stauv2': STAUv2.RNN, 
            # 'aauv2': AAUv2.RNN
        }
        num_hidden = []
        for i in range(configs.num_layers):
            num_hidden.append(configs.num_hidden)
        self.num_hidden = num_hidden
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        # print("Network state:")
        # for param_tensor in self.network.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        #     print(param_tensor, '\t', self.network.state_dict()[param_tensor].size())
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=configs.lr_decay)

        self.MSE_criterion = nn.MSELoss()
        self.L1_loss = nn.L1Loss()

    def save(self, itr):
        stats = {'net_param': self.network.state_dict()}
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save predictive model to %s" % checkpoint_path)

    def load(self, pm_checkpoint_path):
        print('load predictive model:', pm_checkpoint_path)
        stats = torch.load(pm_checkpoint_path, map_location=torch.device(self.configs.device))
        self.network.load_state_dict(stats['net_param'])

    def train(self, data, mask, itr):
        frames = data
        self.network.train()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        next_frames = self.network(frames_tensor, mask_tensor)
        ground_truth = frames_tensor

        batch_size = next_frames.shape[0]

        self.optimizer.zero_grad()
        loss_l1 = self.L1_loss(next_frames,
                               ground_truth[:, 1:])
        loss_l2 = self.MSE_criterion(next_frames,
                                     ground_truth[:, 1:])

        hss_csi_score = HSS_CSI_Score(next_frames, ground_truth[:, 1:])

        loss_gen = loss_l2 + 0.005 * nn.Sigmoid(hss_csi_score)
        loss_gen.backward()
        self.optimizer.step()

        if itr >= self.configs.sampling_stop_iter and itr % self.configs.delay_interval == 0:
            self.scheduler.step()
            # self.scheduler_F.step()
            # self.scheduler_D.step()
            print('Lr decay to:%.8f', self.optimizer.param_groups[0]['lr'])
        return next_frames, loss_l1.detach().cpu().numpy(), loss_l2.detach().cpu().numpy()

    def test(self, data, mask):
        frames = data
        self.network.eval()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()


def HSS_CSI_Score(x, gx):
    save_sample = 5
    assert x.shape[1] == gx.shape[1]
    output_length = x.shape[1]
    batch_size = x.shape[0]

    img_gen = x.permute(0, 1, 3, 4, 2)  # * 0.5 + 0.5
    test_ims = gx.permute(0, 1, 3, 4, 2)  # * 0.5 + 0.5
    img_out = img_gen[:, -output_length:, :]
    test_ims = test_ims[:, -output_length:, :]
    avg_hss20 = 0
    avg_hss35 = 0
    avg_hss45 = 0

    avg_csi20 = 0
    avg_csi35 = 0
    avg_csi45 = 0

    # img_mse, img_mae, img_psnr, ssim, img_lpips, mse_list, mae_list, psnr_list, ssim_list, lpips_list = [], [], [], [], [], [], [], [], [], []
    # img_hss20, img_csi20, HSS_list20, CSI_list20 = [], [], [], []
    # img_hss35, img_csi35, HSS_list35, CSI_list35 = [], [], [], []
    # img_hss45, img_csi45, HSS_list45, CSI_list45 = [], [], [], []
    # for i in range(output_length):
    # img_hss20.append(0)
    # img_hss35.append(0)
    # img_hss45.append(0)
    # img_csi20.append(0)
    # img_csi35.append(0)
    # img_csi45.append(0)
    #
    # HSS_list20.append(0)
    # HSS_list35.append(0)
    # HSS_list45.append(0)
    # CSI_list20.append(0)
    # CSI_list35.append(0)
    # CSI_list45.append(0)

    for i in range(output_length):
        x = test_ims[:, i, :]
        gx = img_out[:, i, :]
        hss20, hss35, hss45 = 0, 0, 0
        csi20, csi35, csi45 = 0, 0, 0
        for id_batch in range(batch_size):
            hssa_20, csia_20 = cal_hss_csi(x, gx, 20, 35)
            hss20 += hssa_20
            csi20 += csia_20

            hssa_35, csia_35 = cal_hss_csi(x, gx, 35, 45)
            hss35 += hssa_35
            csi35 += csia_35

            hssa_45, csia_45 = cal_hss_csi(x, gx, 45, 70)
            hss45 += hssa_45
            csi45 += csia_45

        hss20, hss35, hss45 = hss20 / batch_size, hss35 / batch_size, hss45 / batch_size
        csi20, csi35, csi45 = csi20 / batch_size, csi35 / batch_size, csi45 / batch_size

        # img_hss20[i] += hss20
        # img_hss35[i] += hss35
        # img_hss45[i] += hss45
        #
        # img_csi20[i] += csi20
        # img_csi35[i] += csi35
        # img_csi45[i] += csi45

        # HSS_list20[i] = hss20
        # HSS_list35[i] = hss35
        # HSS_list45[i] = hss45
        # CSI_list20[i] = csi20
        # CSI_list35[i] = csi35
        # CSI_list45[i] = csi45

        avg_hss20 += hss20
        avg_hss35 += hss35
        avg_hss45 += hss45
        avg_csi20 += csi20
        avg_csi35 += csi35
        avg_csi45 += csi45
    avg_hss20, avg_hss35, avg_hss45 = avg_hss20 / output_length, avg_hss35 / output_length, avg_hss45 / output_length

    avg_csi20, avg_csi35, avg_csi45 = avg_csi20 / output_length, avg_csi35 / output_length, avg_csi45 / output_length
    hss_csi_score = ((avg_hss20 + avg_csi20) * 0.5) * 0.25 + ((avg_hss35 + avg_csi35) * 0.5) * 0.35 + (
            (avg_hss45 + avg_csi45) * 0.5) * 0.4
    return hss_csi_score


def HSS(hits, misses, falsealarms, correctnegatives):
    '''
    HSS - Heidke skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: HSS value
    '''

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses ** 2 + falsealarms ** 2 + 2 * hits * correctnegatives +
               (misses + falsealarms) * (hits + correctnegatives))
    if HSS_den == 0:
        return 0.0
    return HSS_num / HSS_den


def CSI(hits, misses, falsealarms, correctnegatives):
    denominator = (hits + falsealarms + misses)
    if (denominator == 0):
        return 0.0
    return hits / denominator


def BIAS(hits, misses, falsealarms, correctnegatives):
    '''
    func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses)
    	  alias: (TP + FP)/(TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    if hits + misses == 0:
        return 0.0
    return (hits + falsealarms) / (hits + misses)


def escore(obs, pre, low, high):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           low=low, high=high)

    hss = HSS(hits, misses, falsealarms, correctnegatives)
    csi = CSI(hits, misses, falsealarms, correctnegatives)

    # bias = BIAS(hits, misses, falsealarms, correctnegatives)
    # fid = calculate_fid(obs, pre)
    # score = hss * (math.exp(-abs(1 - bias)) ** 0.2) * (math.exp(-(fid / 100)) ** 0.2)
    score = hss * csi
    return score


def cal_hss_csi(obs, pre, low, high):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           low=low / 70.0, high=high / 70.0)
    hss = HSS(hits, misses, falsealarms, correctnegatives)
    csi = CSI(hits, misses, falsealarms, correctnegatives)
    return hss, csi


def radar_sc(image1, image2):
    return escore(image1, image2, 20, 35) * 0.3 + escore(image1, image2, 35, 45) * 0.3 + escore(image1, image2, 45,
                                                                                                70) * 0.4


def prep_clf(obs, pre, low=0, high=70):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    '''
    # 根据阈值分类为 0, 1
    obs = (torch.where(obs >= low, 1, 0)) & (torch.where(obs < high, 1, 0))
    pre = (torch.where(pre >= low, 1, 0)) & (torch.where(pre < high, 1, 0))
    # True positive (TP)
    hits = torch.sum((obs == 1) & (pre == 1)).type(torch.float64)
    # False negative (FN)
    misses = torch.sum((obs == 1) & (pre == 0)).type(torch.float64)
    # False positive (FP)
    falsealarms = torch.sum((obs == 0) & (pre == 1)).type(torch.float64)
    # True negative (TN)
    correctnegatives = torch.sum((obs == 0) & (pre == 0)).type(torch.float64)

    return hits, misses, falsealarms, correctnegatives
