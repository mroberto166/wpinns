import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as mc
import colorsys
from scipy.special import legendre
import sobol_seq


class EquationBaseClass:
    def __init__(self, norm, cutoff, weak_form, p=2):
        self.norm = norm
        self.cutoff = cutoff
        self.weak_form = weak_form
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.p = p


    def return_norm(self, func, func_x):
        if self.norm == "H1":
            norm = torch.mean(torch.abs(func) ** self.p + torch.abs(func_x) ** self.p) ** (1 / self.p)
        elif self.norm == "L2":
            norm = torch.mean(torch.abs(func) ** self.p) ** (1 / self.p)
        elif self.norm == "H1s":
            norm = torch.mean(torch.abs(func_x) ** self.p) ** (1 / self.p)
        elif self.norm is None:
            norm = 1
        else:
            raise ValueError()
        return norm


    def fun_w(self, x, extrema_values, time_dimensions):
        if self.cutoff == "def_max":
            dim = x.shape[1]
            I1 = 1
            x_mod = torch.zeros_like(x)
            x_mod_2 = torch.zeros_like(x)

            for i in range(time_dimensions, dim):
                up = extrema_values[i, 1]
                low = extrema_values[i, 0]
                h_len = (up - low) / 2.0
                x_mod[:, i] = (x[:, i] - low - h_len) / h_len

            for i in range(time_dimensions, dim):
                supp_x = torch.gt(torch.tensor(1.0) - torch.abs(x_mod[:, i]), 0)
                x_mod_2[:, i] = torch.where(supp_x, torch.exp(torch.tensor(1.0) / (x_mod[:, i] ** 2 - 1)) / I1, torch.zeros_like(x_mod[:, i]))
            w = 1.0
            for i in range(time_dimensions, dim):
                w = w * x_mod_2[:, i]

            return w / np.max(w.cpu().detach().numpy())
        if self.cutoff == "def_av":
            x.requires_grad = True
            dim = x.shape[1]
            I1 = 0.210987

            x_mod = torch.zeros_like(x)
            x_mod_2 = torch.zeros_like(x)

            for i in range(time_dimensions, dim):
                up = extrema_values[i, 1]
                low = extrema_values[i, 0]
                h_len = (up - low) / 2.0
                x_mod[:, i] = (x[:, i] - low - h_len) / h_len

            for i in range(time_dimensions, dim):
                supp_x = torch.gt(torch.tensor(1.0) - torch.abs(x_mod[:, i]), 0)
                x_mod_2[:, i] = torch.where(supp_x, torch.exp(torch.tensor(1.0) / (x_mod[:, i] ** 2 - 1)) / I1, torch.zeros_like(x_mod[:, i]))
            w = 1.0
            for i in range(time_dimensions, dim):
                w = w * x_mod_2[:, i]
            return w
        elif self.cutoff == "net":
            w = torch.load("EnsDist/Setup_25/Retrain_4/ModelInfo.pkl")
            for param in w.parameters():
                param.requires_grad = False
            x = (x - extrema_values[:, 0]) / (extrema_values[:, 1] - extrema_values[:, 0])
            return w(x)
        if self.cutoff == "quad":
            dim = x.shape[1]
            x_mod = torch.zeros_like(x)

            for i in range(time_dimensions, dim):
                up = extrema_values[i, 1]
                low = extrema_values[i, 0]
                h_len = (up - low) / 2.0
                x_mod[:, i] = (x[:, i] - low - h_len) / h_len

            w = 1.0
            for i in range(time_dimensions, dim):
                w = w * (-x_mod[:, i] ** 2 + 1)

            return w / np.max(w.cpu().detach().numpy())
        else:
            raise ValueError

    def convert(self, vector, extrema_values):
        vector = np.array(vector)
        max_val = np.max(np.array(extrema_values), axis=1)
        min_val = np.min(np.array(extrema_values), axis=1)
        vector = vector * (max_val - min_val) + min_val
        return torch.from_numpy(vector).type(torch.FloatTensor)

    def lighten_color(self, color, amount=0.5):
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
