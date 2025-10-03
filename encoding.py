class SNN:
    def __init__(self):
        # ======= #
        # 定义常量 #
        # ======= #
        encode_batch = 10000
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
        snn_code = np.empty((self.encode_batch, 3, self.N_E + self.N_I))
        for i, X in enumerate(["R", "G", "B"]):  # "R", "G", "B"
            net = self.networks[X]
            for h in range(self.encode_batch):
                print(h)
                net.restore()
                br2.defaultclock.dt = 0.02 * ms
                intensity = self.sample_matrices[X] @ image[h, i, :, :].reshape(-1)
                pr_E, pr_I = self.convert_intensity_to_poisson_rates(intensity)
                net["PG_E"].rates = pr_E
                net["PG_I"].rates = pr_I
                net.run(400 * ms)
                net["PG_E"].rates = (15 * self.K / 5) * Hz
                net["PG_I"].rates = (12 * self.K / 5) * Hz
                net.run(100 * ms)
                snn_code[h, i, :self.N_E] = net["SM_E"].count * 2
                snn_code[h, i, self.N_E:] = net["SM_I"].count * 2
        return snn_code