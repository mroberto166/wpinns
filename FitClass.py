import copy
import torch
import torch.nn as nn
import numpy as np


def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, Ec, network_sol, network_test, x_u_train, u_train, x_b_train, u_b_train, x_f_train, epc, verbose=False, minimizing=True):
        lambda_res = network_sol.lambda_residual
        lambda_reg_sol = network_sol.regularization_param
        lambda_reg_test = network_test.regularization_param

        u_pred_var_list = list()
        u_train_var_list = list()

        Ec.apply_bc(network_sol, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)
        if x_u_train.shape[0] != 0:
            Ec.apply_ic(network_sol, x_u_train, u_train, u_pred_var_list, u_train_var_list)

        u_pred_tot_vars = torch.cat(u_pred_var_list, 0).to(Ec.device)
        u_train_tot_vars = torch.cat(u_train_var_list, 0).to(Ec.device)

        loss_vars = torch.mean(torch.abs(u_pred_tot_vars - u_train_tot_vars) ** Ec.p)

        loss_reg_sol = regularization(network_sol, 2)
        loss_reg_test = regularization(network_test, 2)

        if minimizing:
            if verbose:
                print("############### MINIMIZING ###############")
            loss_pde, loss_pde_no_norm = Ec.compute_res(network_sol, network_test, x_f_train, minimizing)
            loss_v = lambda_res * loss_vars.to(Ec.device) + loss_pde.to(Ec.device) + lambda_reg_sol * loss_reg_sol.to(Ec.device) + lambda_reg_test * loss_reg_test.to(Ec.device)
            if verbose:
                print("###############################################################################################################")
                print("Function Loss    : ", (loss_vars ** (1 / Ec.p)).detach().cpu().numpy(),
                      "\nPDE Residual     : ", (loss_pde_no_norm ** (1 / Ec.p)).detach().cpu().numpy())
                print()
                print()

            return loss_v, loss_vars, loss_pde, loss_pde_no_norm
        else:
            if verbose:
                print("############### MAXIMIZING ###############")
            loss_pde, loss_pde_no_norm = Ec.compute_res(network_sol, network_test, x_f_train, minimizing)

            loss_v = - torch.log(loss_pde)
            return loss_v, 0, 0, [0, 0, 0], loss_pde_no_norm


def fit(Ec, solution_model, test_function_model, optimizer_min, optimizer_max, training_set_class, verbose=False):
    num_epochs = solution_model.num_epochs
    iterations_max = test_function_model.iterations
    iterations_min = solution_model.iterations
    reset_freq = int(solution_model.reset_freq * num_epochs)

    best_losses = list([0, 0, 0, 0, 0])
    freq = 1000

    training_coll = training_set_class.data_coll
    training_boundary = training_set_class.data_boundary
    training_initial_internal = training_set_class.data_initial_internal

    solution_model.train()
    test_function_model.train()

    if iterations_min != 0:
        best_train = 1e+12
    else:
        best_train = 0
    best_solution_model = None
    best_test_function_model = None

    lambda1 = lambda e: 1 / (1 + (e / num_epochs))
    my_lr_scheduler_min = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_min, lr_lambda=lambda1)
    my_lr_scheduler_max = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_max, lr_lambda=lambda1)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        if epoch % reset_freq == 0 and epoch != 0:
            print("Resetting Params")
            test_function_model.apply(weight_reset)

        current_losses = list([0, 0, 0, 0, 0])

        def closure_max():
            optimizer_max.zero_grad()
            loss_test, _, _, _, res_pde_no_norm \
                = CustomLoss().forward(Ec, solution_model, test_function_model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, epoch, verbose, False)
            current_losses[4] = current_losses[4] + float(res_pde_no_norm.cpu().detach().numpy())
            loss_test.backward()
            return loss_test

        def closure_min():
            optimizer_min.zero_grad()
            loss_sol, loss_vars, loss_int, res_pde_no_norm = CustomLoss().forward(Ec, solution_model, test_function_model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, epoch, verbose, True)
            current_losses[0] = current_losses[0] + float(loss_sol.cpu().detach().numpy())
            current_losses[1] = current_losses[1] + float(loss_vars.cpu().detach().numpy())
            current_losses[2] = current_losses[2] + float(loss_int.cpu().detach().numpy())
            current_losses[3] = current_losses[3] + float(res_pde_no_norm.cpu().detach().numpy())

            loss_sol.backward()
            return loss_sol

        if epoch % freq == 0:
            print("##################################################  ", epoch, "  ##################################################")

        batch = 0
        if len(training_boundary) != 0 and len(training_initial_internal) == 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_boundary)):

                x_u_train_ = torch.full((0, 1), 0)
                u_train_ = torch.full((0, 1), 0)

                for _ in range(iterations_max):
                    optimizer_max.step(closure=closure_max)

                for _ in range(iterations_min):
                    optimizer_min.step(closure=closure_min)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()
                batch = batch + 1

        if len(training_boundary) != 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_u_train_, u_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_initial_internal, training_boundary)):
                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                for _ in range(iterations_max):
                    optimizer_max.step(closure=closure_max)

                for _ in range(iterations_min):
                    optimizer_min.step(closure=closure_min)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

                batch = batch + 1
        for l in range(len(current_losses)):
            current_losses[l] = current_losses[l] / batch

        if np.isnan(current_losses[0]):
            print("WARNING: Found NaN")
            return best_losses, best_solution_model, best_test_function_model

        if current_losses[0] < best_train:
            best_solution_model = copy.deepcopy(solution_model)
            # best_test_function_model = copy.deepcopy(test_function_model)
            best_losses[0] = current_losses[0]
            best_losses[1] = current_losses[1]
            best_losses[2] = current_losses[2]
            best_losses[3] = current_losses[3]
            best_losses[4] = current_losses[4]
            best_train = current_losses[0]

        my_lr_scheduler_min.step()
        my_lr_scheduler_max.step()
    return best_losses, best_solution_model, best_test_function_model



