from EquationModels.ShockRarEntropy import EquationClass
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


def fun_w(x):
    dim = x.shape[1]
    I1 = 1
    x_mod = torch.zeros_like(x)
    x_mod_2 = torch.zeros_like(x)

    for i in range(1, dim):
        up = 1
        low = -1
        h_len = (up - low) / 2.0
        x_mod[:, i] = (x[:, i] - low - h_len) / h_len

    for i in range(1, dim):
        supp_x = torch.gt(torch.tensor(1.0) - torch.abs(x_mod[:, i]), 0)
        x_mod_2[:, i] = torch.where(supp_x, torch.exp(torch.tensor(1.0) / (x_mod[:, i] ** 2 - 1)) / I1, torch.zeros_like(x_mod[:, i]))
    w = 1.0
    for i in range(1, dim):
        w = w * x_mod_2[:, i]

    return w / np.max(w.cpu().detach().numpy())


base_path_list = ["RarefactionWave/Setup_39"]
des = "rar"
ec = EquationClass(None, None, None, None)

gauss = False
sine = False
N_s = 10

exp = 1
plot = True
test_ent = False
final_T = False

if not gauss:
    file_ex = "Data/BurgersExact.txt"
    exact_solution = np.loadtxt(file_ex)
    inputs = torch.from_numpy(exact_solution[np.where((np.isclose(exact_solution[:, 0], 0, 1e-3)) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2])
    inputs = inputs.reshape(inputs.shape[1], 2)
else:
    file_ex = "Data/DataGauss.txt"
    exact_solution = np.loadtxt(file_ex)
    inputs = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], 0.75, 2e-3)), :-1]).type(torch.float32)
    inputs = inputs.reshape(inputs.shape[1], 2)
if sine:
    time_steps = [0.0, 0.25, 0.5, 0.75]
    time = 1
if gauss:
    time_steps = [0.0018, 0.25, 0.5, 0.75]
    time = 0.75
if not sine and not gauss:
    time_steps = [0.0, 0.25, 0.45]
    time = 0.4
scale_vec = np.linspace(0.65, 1.55, len(time_steps))

N_s_list = list([N_s])
best_err = list()
average_err = list()

for N_s in N_s_list:
    solution_u = torch.zeros((N_s, len(time_steps), inputs.shape[0]))
    solution_test = torch.zeros((N_s, len(time_steps), inputs.shape[0]))
    solution_u_ex = torch.zeros((N_s, len(time_steps), inputs.shape[0]))
    average_solution = 0
    best_solution = 0
    best_model_idx_train = None
    smallest_train_err = 10

    if test_ent:
        average_test_ent = 0
        solution_test_ent = torch.zeros((N_s, len(time_steps), inputs.shape[0]))
    for base_path in base_path_list:
        if not gauss:
            if final_T:
                inputs = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], time, 1e-3) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2]).type(
                    torch.FloatTensor)
            else:
                inputs = torch.from_numpy(exact_solution[np.where((exact_solution[:, 0] < time) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2]).type(
                    torch.FloatTensor)
        else:
            if final_T:
                inputs = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], time, 2e-3)), :-1]).type(torch.float32)
            else:
                inputs = torch.from_numpy(exact_solution[np.where(exact_solution[:, 0] < time), :-1]).type(torch.float32)
        inputs = inputs.reshape(inputs.shape[1], 2)
        print("#################################################")

        directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        print(directories_model)

        for i, retrain_path in enumerate(directories_model):
            if os.path.isfile(base_path + "/" + retrain_path + "/InfoModel.txt"):
                print(base_path + "/" + retrain_path + "/InfoModel.txt ")
                info_model = pd.read_csv(base_path + "/" + retrain_path + "/InfoModel.txt", header=None, sep=",", index_col=0)
                info_model = info_model.transpose().reset_index().drop("index", 1)
                info_model["metric"] = info_model["loss_pde_no_norm"] + info_model["loss_vars"]
                pickle_inn_sol = open(base_path + "/" + retrain_path + "/ModelSol.pkl", 'rb')

                mod_sol = torch.load(pickle_inn_sol)

                outputs_sol = mod_sol(inputs)[:, 0].detach()
                if test_ent:
                    pickle_inn_test_ent = open(base_path + "/" + retrain_path + "/ModelEntropy.pkl", 'rb')
                    mod_test_ent = torch.load(pickle_inn_test_ent)

                    outputs_test_ent = (mod_test_ent(inputs)[:, 0].detach()) ** exp * fun_w(inputs)

                if i < N_s:
                    average_solution = average_solution + outputs_sol / N_s

                    print("##############################")
                    print(retrain_path)
                    print(info_model["rel_L2_norm"].values)
                    print(info_model["metric"].values)

                    if info_model["metric"].values < smallest_train_err:
                        best_model_idx_train = i
                        best_solution = outputs_sol
                        smallest_train_err = info_model["metric"].values

                for k, val in enumerate(time_steps):
                    if not gauss:

                        inputs_val = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 1e-3) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2]).type(torch.FloatTensor)
                    else:
                        inputs_val = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 2e-3)), :-1]).type(torch.float32)
                    inputs_val = inputs_val.reshape(inputs_val.shape[1], 2)
                    outputs_u = mod_sol(inputs_val)[:, 0].detach()
                    solution_u[i, k, :] = outputs_u
            else:
                print(base_path + "/" + retrain_path + "/InfoModel.txt not found")

    best_train_mod_u = solution_u[best_model_idx_train, :, :]
    best_train_mod_test = solution_test[best_model_idx_train, :, :]

    mean_u = torch.mean(solution_u, 0)
    std_u = torch.std(solution_u, 0)

    if not sine and not gauss:
        ex = ec.exact(inputs).detach().numpy()
    if sine:
        if final_T:
            ex = exact_solution[np.where(np.isclose(exact_solution[:, 0], time, 1e-3) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), -1]
        else:
            ex = exact_solution[np.where((exact_solution[:, 0] < time) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), -1]

    if gauss:
        if final_T:
            ex = exact_solution[np.where(np.isclose(exact_solution[:, 0], time, 2e-3)), -1]
        else:
            ex = exact_solution[np.where(exact_solution[:, 0] < time), -1]
        ex = ex.reshape(-1, )

    average_l1_err = np.mean(abs(ex.reshape(-1, ) - average_solution.detach().numpy().reshape(-1, ))) / np.mean(abs(ex.reshape(-1, )))
    average_l2_err = (np.mean(abs(ex.reshape(-1, ) - average_solution.detach().numpy().reshape(-1, )) ** 2) / np.mean(abs(ex.reshape(-1, )) ** 2)) ** 0.5
    average_max_err = np.max(abs(ex.reshape(-1, ) - average_solution.detach().numpy().reshape(-1, ))) / np.max(abs(ex.reshape(-1, )))

    best_l1_err = np.mean(abs(ex.reshape(-1, ) - best_solution.detach().numpy()).reshape(-1, )) / np.mean(abs(ex.reshape(-1, )))
    best_l2_err = (np.mean(abs(ex.reshape(-1, ) - best_solution.detach().numpy().reshape(-1, )) ** 2) / np.mean(abs(ex.reshape(-1, )) ** 2)) ** 0.5
    print("Average L1 error:", average_l1_err)
    print("Average L2 error:", average_l2_err)
    print("Average Linf error:", average_max_err)
    print("Best Trained L1 error:", best_l1_err)
    print("Best Trained L2 error:", best_l2_err)
    average_err.append(average_l1_err)
    if plot:

        p = 1

        fig = plt.figure()
        plt.grid(True, which="both", ls=":")
        for k, (val, scale) in enumerate(zip(time_steps, scale_vec)):
            if not gauss:
                inputs = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 1e-3) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2]).type(torch.FloatTensor)
            else:
                inputs = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 2e-3)), :-1]).type(torch.float32)
            inputs = inputs.reshape(inputs.shape[1], 2)
            x_plot = inputs[:, 1].reshape(-1, 1)

            if not sine and not gauss:
                ex = ec.exact(inputs)
            if sine:
                ex = exact_solution[np.where((exact_solution[:, 0] == val) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), -1].reshape(-1, 1)
            if gauss:
                ex = exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 2e-3)), -1]
                ex = ex.reshape(-1, )

            plt.plot(x_plot.cpu().detach().numpy(), ex, linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$', color=ec.lighten_color('grey', scale), zorder=0)
            plt.plot(x_plot.cpu().detach().numpy(), mean_u[k, :], label=r'Predicted, $t=$' + str(val) + r'$s$', color=ec.lighten_color('C0', scale), zorder=10)
            plt.fill_between(x_plot.cpu().detach().numpy().reshape(-1, ), mean_u[k, :] - 2 * std_u[k, :], mean_u[k, :] + 2 * std_u[k, :], alpha=0.25, color="grey")

        plt.xlabel(r'$x$')
        plt.ylabel(r'$u$')
        plt.legend()
        plt.savefig(base_path_list[0] + "/u_average_" + des + ".png", dpi=500)

        fig = plt.figure()
        plt.grid(True, which="both", ls=":")
        for k, (val, scale) in enumerate(zip(time_steps, scale_vec)):

            if not gauss:
                inputs = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 1e-3) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2]).type(torch.FloatTensor)
            else:
                inputs = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 2e-3)), :-1]).type(torch.float32)
            inputs = inputs.reshape(inputs.shape[1], 2)
            x_plot = inputs[:, 1].reshape(-1, 1)

            if not sine and not gauss:
                ex = ec.exact(inputs)
            if sine:
                ex = exact_solution[np.where((exact_solution[:, 0] == val) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), -1].reshape(-1, 1)
            if gauss:
                ex = exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 2e-3)), -1]
                ex = ex.reshape(-1, )

            plt.plot(x_plot.cpu().detach().numpy(), ex, linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$', color=ec.lighten_color('grey', scale), zorder=0)
            plt.plot(x_plot.cpu().detach().numpy(), best_train_mod_u[k, :], label=r'Predicted, $t=$' + str(val) + r'$s$', color=ec.lighten_color('C0', scale), zorder=10)

        plt.xlabel(r'$x$')
        plt.ylabel(r'$u$')
        plt.legend()
        plt.savefig(base_path_list[0] + "/u_best_trained_" + des + ".png", dpi=500)
