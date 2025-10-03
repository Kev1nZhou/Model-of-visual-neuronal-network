## 导入区
import traceback
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import brian2 as br2
from brian2.units import ms, Hz

## 模型定义区

class SNN:
    def __init__(self):
        # ======= #
        # 定义常量 #
        # ======= #
        encode_batch = 1
        N_E = 500
        N_I = 500
        V_r = 0.0
        V_t = 1.0
        T_tau = 20.0 * ms
        T_refractory = 1.0 * ms
        K = 50
        J_in = 1.0 * K ** -0.5
        J_EE = J_in * 1.0
        J_IE = J_in * 1.0
        J_II = J_in * 1.8
        J_EI = J_in * 2.0
        J_EP = J_EE
        J_IP = J_IE
        Image_W = 32
        Image_H = 32
        # =========== #
        # 加载采样矩阵 #
        # =========== #
        sample_matrices = self.load_sparse_matrices(
            "./Model/sparse_sample_matrices_1000_10.npz"
        )

        __shape = (N_E + N_I, Image_W * Image_H)
        R_shape = sample_matrices["R"].shape
        G_shape = sample_matrices["G"].shape
        B_shape = sample_matrices["B"].shape
        assert R_shape == __shape, f"R shape is {R_shape}, expected {__shape}"
        assert G_shape == __shape, f"R shape is {G_shape}, expected {__shape}"
        assert B_shape == __shape, f"R shape is {B_shape}, expected {__shape}"
        # =========== #
        # 加载邻接矩阵 #
        # =========== #
        adjacency_matrices = np.load('./Model/sparse_rgbCEE_adjacency_matrices_1000_50.npz',
                                     allow_pickle=True)
        ini_voltage = np.load('./Model/ini_voltage_1000_50.npz.npy')
        # =============== #
        # 定义 brian2 模型 #
        # =============== #
        networks = {}
        for X, C in zip(["R", "G", "B"], range(3)):
            br2.start_scope()

            neuron_params = dict(
                model="dV/dt = (V_r - V) / T_tau : 1  (unless refractory)",
                threshold="V > V_t",
                reset="V = V_r",
                refractory=T_refractory,
                method="euler",
                namespace={
                    "V_r": V_r,
                    "V_t": V_t,
                    "T_tau": T_tau,
                },
            )
            NG_E = br2.NeuronGroup(N_E, name="NG_E", **neuron_params)
            NG_I = br2.NeuronGroup(N_I, name="NG_I", **neuron_params)
            NG_E.V = ini_voltage[0, C, :N_E]
            NG_I.V = ini_voltage[0, C, N_E:]
            PG_E = br2.PoissonGroup(N_E, rates=0 * Hz, name="PG_E")
            PS_E = br2.Synapses(PG_E, NG_E, "w : 1", on_pre="V += w")
            PS_E.connect(j="i")
            PS_E.w = J_EP

            PG_I = br2.PoissonGroup(N_I, rates=0 * Hz, name="PG_I")
            PS_I = br2.Synapses(PG_I, NG_I, "w : 1", on_pre="V += w")
            PS_I.connect(j="i")
            PS_I.w = J_IP

            Ad = adjacency_matrices[X].item()
            targets, sources = Ad["C_EE"].nonzero()
            print('第{X}通道行中0个数为：', np.sum(targets == 0))
            S_EE = br2.Synapses(NG_E, NG_E, "w : 1", on_pre="V += w")
            S_EE.connect(i=sources, j=targets)
            S_EE.w = J_EE

            targets, sources = Ad["C_IE"].nonzero()
            S_IE = br2.Synapses(NG_E, NG_I, "w : 1", on_pre="V += w")
            S_IE.connect(i=sources, j=targets)
            S_IE.w = J_IE

            targets, sources = Ad["C_EI"].nonzero()
            S_EI = br2.Synapses(NG_I, NG_E, "w : 1", on_pre="V = V -  w")
            S_EI.connect(i=sources, j=targets)
            S_EI.w = J_EI

            targets, sources = Ad["C_II"].nonzero()
            S_II = br2.Synapses(NG_I, NG_I, "w : 1", on_pre="V = V - w")
            S_II.connect(i=sources, j=targets)
            S_II.w = J_II

            SM_E = br2.SpikeMonitor(NG_E, name="SM_E")
            SM_I = br2.SpikeMonitor(NG_I, name="SM_I")

            net = br2.Network(
                NG_E, NG_I,
                PG_E, PS_E, PG_I, PS_I,
                S_EE, S_IE, S_EI, S_II,
                SM_E, SM_I
            )
            net.store()
            networks[X] = net

        # =========== #
        # 存储必要数据 #
        # =========== #

        self.networks = networks
        self.sample_matrices = sample_matrices
        self.N_E = N_E
        self.N_I = N_I
        self.K = K
        self.Image_W = Image_W
        self.Image_H = Image_H
        self.encode_batch = encode_batch
        self.J_EE = J_EE
        self.J_IE = J_IE
        self.J_EI = J_EI
        self.J_II = J_II
        self.J_EP = J_EP
        self.J_IP = J_IP
        self.T_refractory = T_refractory

    def load_sparse_matrices(self, path: str):
        d = np.load(path)
        return {
            "R": sp.csr_matrix((d["R_data"], (d["R_row"], d["R_col"])), d["R_shape"]),
            "G": sp.csr_matrix((d["G_data"], (d["G_row"], d["G_col"])), d["G_shape"]),
            "B": sp.csr_matrix((d["B_data"], (d["B_row"], d["B_col"])), d["B_shape"]),
        }

    def convert_intensity_to_poisson_rates(self, intensity):
        N_E = self.N_E
        K = self.K
        W = self.Image_W
        H = self.Image_H
        pr_E = (15 * K / 5) * (1.0 + 6 * intensity[:N_E] / (W * H)) * Hz
        pr_I = (12 * K / 5) * (1.0 + 6 * intensity[N_E:] / (W * H)) * Hz
        return pr_E, pr_I

    def encode(self, image):  # image.shape   统一[B,C,W,H]
        image = image.transpose(2, 0, 1)
        print("image.shape =", image.shape)
        snn_code = np.empty((3, self.N_E + self.N_I))
        for i, X in enumerate(["R", "G", "B"]):  # "R", "G", "B"
            net = self.networks[X]
            net.restore()
            br2.defaultclock.dt = 0.02 * ms
            intensity = self.sample_matrices[X] @ image[i,:,:].reshape(-1)
            pr_E, pr_I = self.convert_intensity_to_poisson_rates(intensity)
            net["PG_E"].rates = pr_E
            net["PG_I"].rates = pr_I
            net.run(400 * ms)
            net["PG_E"].rates = (15 * self.K / 5) * Hz
            net["PG_I"].rates = (12 * self.K / 5) * Hz
            net.run(100 * ms)
            snn_code[i, :self.N_E] = net["SM_E"].count * 2
            snn_code[i, self.N_E:] = net["SM_I"].count * 2
        return snn_code


class DenseDecoder(nn.Module):
    def __init__(self):
        # ======= #
        # 定义常量 #
        # ======= #
        super(DenseDecoder, self).__init__()
        N_E = 500
        N_I = 500
        self.Image_W = 32
        self.Image_H = 32
        # ======= #
        # 定义模型 #
        # ======= #
        self.fc1 = nn.Linear(3000, 3 * 1024)
        self.bn1 = nn.BatchNorm1d(3 * 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024 * 3, 3 * 32 * 32)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        # print(x.size())
        x = x.view(-1,3,32, 32)
        return x


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)  # [b,3,32,32]
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)  # [b,64,16,16]

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)  # [b,128,8,8]

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(1024)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(1024)  # [b,1024,8,8]

        self.up6 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv6_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(512)
        self.conv6_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(512)

        self.up7 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv7_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(256)
        self.conv7_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(256)

        self.up8 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn8_1 = nn.BatchNorm2d(128)
        self.conv8_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn8_2 = nn.BatchNorm2d(128)

        self.up9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn9_2 = nn.BatchNorm2d(64)
        self.conv9_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn9_3 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        conv1 = self.conv1_1(x)
        conv1 = self.bn1_1(conv1)
        conv1 = F.relu(conv1)
        conv1 = self.conv1_2(conv1)
        conv1 = self.bn1_2(conv1)
        conv1 = F.relu(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2_1(pool1)
        conv2 = self.bn2_1(conv2)
        conv2 = F.relu(conv2)
        conv2 = self.conv2_2(conv2)
        conv2 = self.bn2_2(conv2)
        conv2 = F.relu(conv2)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3_1(pool2)
        conv3 = self.bn3_1(conv3)
        conv3 = F.relu(conv3)
        conv3 = self.conv3_2(conv3)
        conv3 = self.bn3_2(conv3)
        conv3 = F.relu(conv3)

        conv4 = self.conv4_1(conv3)
        conv4 = self.bn4_1(conv4)
        conv4 = F.relu(conv4)
        conv4 = self.conv4_2(conv4)
        conv4 = self.bn4_2(conv4)
        conv4 = F.relu(conv4)

        conv5 = self.conv5_1(conv4)
        conv5 = self.bn5_1(conv5)
        conv5 = F.relu(conv5)
        conv5 = self.conv5_2(conv5)
        conv5 = self.bn5_2(conv5)
        conv5 = F.relu(conv5)

        up6 = self.up6(conv5)
        merge6 = torch.cat((conv4, up6), dim=1)
        conv6 = self.conv6_1(merge6)
        conv6 = self.bn6_1(conv6)
        conv6 = F.relu(conv6)
        conv6 = self.conv6_2(conv6)
        conv6 = self.bn6_2(conv6)
        conv6 = F.relu(conv6)

        up7 = self.up7(conv6)
        merge7 = torch.cat((conv3, up7), dim=1)
        conv7 = self.conv7_1(merge7)
        conv7 = self.bn7_1(conv7)
        conv7 = F.relu(conv7)
        conv7 = self.conv7_2(conv7)
        conv7 = self.bn7_2(conv7)
        conv7 = F.relu(conv7)

        up8 = self.up8(conv7)
        up8 = F.interpolate(up8, size=(16, 16), mode="bilinear", align_corners=False)
        merge8 = torch.cat((conv2, up8), dim=1)
        conv8 = self.conv8_1(merge8)
        conv8 = self.bn8_1(conv8)
        conv8 = F.relu(conv8)
        conv8 = self.conv8_2(conv8)
        conv8 = self.bn8_2(conv8)
        conv8 = F.relu(conv8)

        up9 = self.up9(conv8)
        up9 = F.interpolate(up9, size=(32, 32), mode="bilinear", align_corners=False)
        merge9 = torch.cat((conv1, up9), dim=1)
        conv9 = self.conv9_1(merge9)
        conv9 = self.bn9_1(conv9)
        conv9 = F.relu(conv9)
        conv9 = self.conv9_2(conv9)
        conv9 = self.bn9_2(conv9)
        conv9 = F.relu(conv9)
        conv9 = self.conv9_3(conv9)
        conv9 = self.bn9_3(conv9)
        conv9 = F.relu(conv9)
        conv10 = self.conv10(conv9)

        return conv10


class End2EndTrain(nn.Module):
    def __init__(self):
        super(End2EndTrain, self).__init__()
        self.dense_decoder = DenseDecoder()
        self.cae = CAE()

    def forward(self, x):
        x = self.dense_decoder(x)
        x = self.cae(x)
        return x


## 关键函数区


def load_snn_model():
    """
    说明: 加载 SNN 模型, 用于对图像进行编码
    输入: 无
    输出:
        + snn_model: SNN 模型
    """
    snn_mode = SNN()
    return snn_mode


def load_cnn_model():
    """
    说明: 加载 CNN 模型, 用于对编码进行解码
    输入:
    输出:
        + cnn_model: CNN 模型
    """
    # cnn_mode = End2EndTrain()
    cnn_model = torch.load("./Model/10101820_100.pth")
    cnn_model.eval()
    return cnn_model


def encode(snn_model, image):
    """
    说明: 用 SNN 模型对图像进行编码
    输入:
        + snn_model: SNN 模型
        + image: 待编码的图像, 以 np.array 存储的 (H, W, 3) 数组
    输出:
        + snn_code: 用 SNN 所得到的编码
    """
    snn_code = snn_model.encode(image)
    return snn_code


def decode(cnn_model, snn_code):
    """
    说明: 用 CNN 模型对编码进行解码
    输入:
        + cnn_model: CNN 模型
        + snn_code: 用 SNN 所得到的编码
    输出:
        + image: 编码后的图像, 以 np.array 存储的 (H, W, 3) 数组
    """
    DEVICE = "cuda"
    cnn_model.to(DEVICE)
    x = torch.FloatTensor(snn_code).to(DEVICE)
    with torch.no_grad():
        x = cnn_model(x.view(1, -1))
        x = torch.clamp(x, 0.0, 1.0)
        x = 255.0 * x
    
    image = x.cpu().squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
    return image


