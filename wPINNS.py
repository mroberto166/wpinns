from EquationModels.ShockRarEntropy import EquationClass

from ModelClass import Pinns, PinnsTest
from FitClass import fit
from DatasetClass import DefineDataset
import os
import sys
import json
import pprint
import torch
import torch.optim as optim
import time

# torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def dump_to_file():
    with open(folder_path + os.sep + "EnsembleInfo.csv", "w") as w:
        keys = list(ensemble_configurations.keys())
        vals = list(ensemble_configurations.values())
        for i in range(0, len(keys)):
            w.write(keys[i] + ", " + str(vals[i]) + "\n")

    with open(folder_path + '/InfoModel.txt', 'w') as file:
        file.write("Nu_train, " + str(N_u_train) + "\n"
                                                   "Nf_train, " + str(N_coll_train) + "\n"
                                                                                      "Nint_train, " + str(N_int_train) + "\n"
                                                                                                                          "validation_size, " + str(validation_size) + "\n"
                                                                                                                                                                       "train_time, " + str(end) + "\n"
                                                                                                                                                                                                   "L2_norm_test, " + str(L2) + "\n"
                                                                                                                                                                                                                                "rel_L2_norm, " + str(L2_rel) + "\n"
                                                                                                                                                                                                                                                                "loss_tot, " + str(best_losses[0]) + "\n"
                                                                                                                                                                                                                                                                                                     "loss_vars, " + str(best_losses[1]) + "\n"
                                                                                                                                                                                                                                                                                                                                           "loss_pde, " + str(best_losses[2]) + "\n"
                                                                                                                                                                                                                                                                                                                                                                                "loss_pde_no_norm, " + str(best_losses[3]) + "\n"
                                                                                                                                                                                                                                                                                                                                                                                                                                       "loss_pde_no_norm_after_max, " + str(best_losses[4]) + "\n"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              "retrain, " + str(
            retrain) + "\n")


def initialize_inputs(len_sys_argv):
    if len_sys_argv == 1:

        # Random Seed for sampling the dataset
        sampling_seed_ = 8

        # Number of training+validation points
        n_coll_ = 16384
        n_u_ = 8192
        n_int_ = 0

        # Additional Info
        folder_path_ = "T4"
        validation_size_ = 0.0
        ensemble_configurations_ = {
            "hidden_layers_sol": 6,
            "hidden_layers_test": 4,

            "neurons_sol": 20,
            "neurons_test": 10,

            "activation_sol": "tanh",
            "activation_test": "tanh",

            "tau_sol": 0.01,
            "tau_test": 0.015,

            "iterations_min": 8,
            "iterations_max": 1,

            "residual_parameter": 10,
            "kernel_regularizer": 2,
            "regularization_parameter_sol": 0.,
            "regularization_parameter_test": 0.,
            "batch_size": (n_coll_ + n_u_ + n_int_),
            "epochs": 100,
            "norm": "H1",
            "cutoff": "def_max",
            "weak_form": "partial",
            "reset_freq": 0.025,
            "loss_type": "l2"
        }

        retrain_ = 31
        shuffle_ = False

    elif len_sys_argv == 10:
        print(sys.argv)
        # Random Seed for sampling the dataset
        sampling_seed_ = int(sys.argv[1])

        # Number of training points
        n_coll_ = int(sys.argv[2])
        n_u_ = int(sys.argv[3])
        n_int_ = int(sys.argv[4])

        # Additional Info
        folder_path_ = sys.argv[5]
        validation_size_ = float(sys.argv[6])
        ensemble_configurations_ = json.loads(sys.argv[7])

        retrain_ = sys.argv[8]
        if sys.argv[9] == "false":
            shuffle_ = False
        else:
            shuffle_ = True
    else:
        raise ValueError("One input is missing")

    return sampling_seed_, n_coll_, n_u_, n_int_, folder_path_, validation_size_, ensemble_configurations_, retrain_, shuffle_


sampling_seed, N_coll, N_u, N_int, folder_path, validation_size, ensemble_configurations, retrain, shuffle = initialize_inputs(len(sys.argv))

norm = None if ensemble_configurations["norm"] == "None" else ensemble_configurations["norm"]
if ensemble_configurations["loss_type"] == "l2":
    loss_type = torch.nn.MSELoss()
elif ensemble_configurations["loss_type"] == "l1":
    loss_type = torch.nn.L1Loss()
elif ensemble_configurations["loss_type"] == "sl1":
    loss_type = torch.nn.SmoothL1Loss()
else:
    raise ValueError

Ec = EquationClass(norm=norm,
                   cutoff=ensemble_configurations["cutoff"],
                   weak_form=ensemble_configurations["weak_form"],
                   p=2)

parameters_values = Ec.parameters_values
space_dimensions = Ec.space_dimensions
time_dimension = Ec.time_dimensions
parameter_dimensions = Ec.parameter_dimensions
output_dimension = Ec.output_dimension
input_dimensions = parameter_dimensions + time_dimension + space_dimensions
extrema = Ec.extrema_values

if Ec.extrema_values is not None:
    print("Using free shape. Make sure you have the functions:")
    print("     - add_boundary(n_samples)")
    print("     - add_collocation(n_samples)")
    print("in the Equation file")

print("######################################")
print("*******Domain Properties********")
print(extrema)
print(input_dimensions)
print(output_dimension)

N_u_train = int(N_u * (1 - validation_size))
N_coll_train = int(N_coll * (1 - validation_size))
N_int_train = int(N_int * (1 - validation_size))
N_train = N_u_train + N_coll_train + N_int_train

if space_dimensions > 0:
    N_b_train = int(N_u_train / (4 * space_dimensions))
else:
    N_b_train = 0
if time_dimension == 1:
    N_i_train = N_u_train - 2 * space_dimensions * N_b_train
elif time_dimension == 0:
    N_b_train = int(N_u_train / (2 * space_dimensions))
    N_i_train = 0
else:
    raise ValueError()

print("\n######################################")
print("*******Info Training Points********")
print("Number of train collocation points: ", N_coll_train)
print("Number of initial and boundary points: ", N_u_train, N_i_train, N_b_train)
print("Number of internal points: ", N_int_train)
print("Total number of training points: ", N_train)

print("\n######################################")
print("*******Ensemble Configurations********")
pprint.pprint(ensemble_configurations)
batch_dim = ensemble_configurations["batch_size"]
epochs = ensemble_configurations["epochs"]
if batch_dim == "full":
    batch_dim = N_train

network_properties_1 = {"hidden_layers": ensemble_configurations["hidden_layers_sol"],
                        "neurons": ensemble_configurations["neurons_sol"],
                        "activation": ensemble_configurations["activation_sol"],
                        "residual_parameter": ensemble_configurations["residual_parameter"],
                        "kernel_regularizer": ensemble_configurations["kernel_regularizer"],
                        "regularization_parameter": ensemble_configurations["regularization_parameter_sol"],
                        "batch_size": batch_dim,
                        "iterations": ensemble_configurations["iterations_min"],
                        "epochs": ensemble_configurations["epochs"],
                        "reset_freq": ensemble_configurations["reset_freq"],
                        "loss_type": loss_type}

network_properties_2 = {"hidden_layers": ensemble_configurations["hidden_layers_test"],
                        "neurons": ensemble_configurations["neurons_test"],
                        "activation": ensemble_configurations["activation_test"],
                        "residual_parameter": ensemble_configurations["residual_parameter"],
                        "kernel_regularizer": ensemble_configurations["kernel_regularizer"],
                        "regularization_parameter": ensemble_configurations["regularization_parameter_test"],
                        "batch_size": batch_dim,
                        "iterations": ensemble_configurations["iterations_max"],
                        "epochs": ensemble_configurations["epochs"],
                        "reset_freq": ensemble_configurations["reset_freq"]}

print("\n######################################")
print("******* Network Properties Solution ********")
pprint.pprint(network_properties_1)

print("\n######################################")
print("******* Network Properties Test ********")
pprint.pprint(network_properties_2)

# ##############################################################################################
# Datasets Creation
print("\n######################################")
print("******* Dimension********")
print("Space Dimensions", space_dimensions)
print("Time Dimensions", time_dimension)
print("Parameter Dimensions", parameter_dimensions)

training_set_class = DefineDataset(Ec, N_coll_train, N_b_train, N_i_train, N_int_train, batches=batch_dim, random_seed=sampling_seed, shuffle=shuffle)
training_set_class.assemble_dataset()

torch.manual_seed(retrain)
solution_model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension, network_properties=network_properties_1)
test_function_model = PinnsTest(input_dimension=input_dimensions, output_dimension=1, network_properties=network_properties_2)
os.mkdir(folder_path)

if torch.cuda.is_available():
    print("Loading model on GPU")
    solution_model.cuda()
    test_function_model.cuda()

print("Fitting Model")
solution_model.train()
test_function_model.train()

sol_params = solution_model.parameters()
test_params = test_function_model.parameters()

optimizer_min = optim.Adam(sol_params, lr=ensemble_configurations["tau_sol"], amsgrad=True)
optimizer_max = optim.Adam(test_params, lr=ensemble_configurations["tau_test"], amsgrad=True)

start = time.time()

best_losses, best_solution_model, best_test_function_model = fit(Ec,
                                                                 solution_model,
                                                                 test_function_model,
                                                                 optimizer_min,
                                                                 optimizer_max,
                                                                 training_set_class,
                                                                 verbose=False)

end = time.time() - start

images_path = folder_path + "/ImagesSol"
os.mkdir(images_path)

L2, L2_rel = Ec.compute_generalization_error(best_solution_model, images_path)

torch.save(best_solution_model, folder_path + "/ModelSol.pkl")
torch.save(best_test_function_model, folder_path + "/ModelTest.pkl")

dump_to_file()
