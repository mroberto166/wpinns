import os
import sys
import itertools

rs = 32
N_coll = int(sys.argv[1])
N_u = int(sys.argv[2])
N_int = int(sys.argv[3])
folder_name = sys.argv[4]
validation_size = 0.0
ensemble_configurations = {
    "hidden_layers_sol": [4, 6],
    "hidden_layers_test": [2, 4],

    "neurons_sol": [20],
    "neurons_test": [10],

    "activation_sol": ["tanh"],
    "activation_test": ["tanh", "sin"],

    "tau_sol": [0.01],
    "tau_test": [0.015],

    "iterations_min": [1],
    "iterations_max": [6, 8],

    "residual_parameter": [10],
    "kernel_regularizer": [2],
    "regularization_parameter_sol": [0],
    "regularization_parameter_test": [0],
    "batch_size": [N_coll + N_u + N_int],
    "epochs": [5000],
    "norm": ["H1"],  # L2, H1, H1s (H1-seminorm)
    "cutoff": ["def_max"],  # This can be left like this
    "weak_form": ["partial"],  # partial, full
    "reset_freq": [0.025, 0.05, 0.25],
    "loss_type": ["l2"],  # l2, l1, sl1 (smooth l1)
}


shuffle = "false"
cluster = sys.argv[5]
GPU = "None"  # GPU="GeForceGTX1080"  # GPU = "GeForceGTX1080Ti"  # GPU = "TeslaV100_SXM2_32GB"
# GPU = "GeForceGTX1080"  # GPU="GeForceGTX1080"  # GPU = "GeForceGTX1080Ti"  # GPU = "TeslaV100_SXM2_32GB"
n_retrain = 10

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
settings = list(itertools.product(*ensemble_configurations.values()))

i = 0
for setup in settings:
    print(setup)

    folder_path = folder_name + "/Setup_" + str(i)
    print("###################################")
    setup_properties = {
        "hidden_layers_sol": setup[0],
        "hidden_layers_test": setup[1],

        "neurons_sol": setup[2],
        "neurons_test": setup[3],

        "activation_sol": setup[4],
        "activation_test": setup[5],

        "tau_sol": setup[6],
        "tau_test": setup[7],

        "iterations_min": setup[8],
        "iterations_max": setup[9],

        "residual_parameter": setup[10],
        "kernel_regularizer": setup[11],
        "regularization_parameter_sol": setup[12],
        "regularization_parameter_test": setup[13],
        "batch_size": setup[14],
        "epochs": setup[15],
        "norm": setup[16],
        "cutoff": setup[17],
        "weak_form": setup[18],
        "reset_freq": setup[19],
        "loss_type": setup[20]
    }

    arguments = list()
    arguments.append(str(rs))
    arguments.append(str(N_coll))
    arguments.append(str(N_u))
    arguments.append(str(N_int))
    arguments.append(str(folder_path))
    arguments.append(str(validation_size))
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        arguments.append("\'" + str(setup_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append(str(setup_properties).replace("\'", "\""))
    arguments.append(str(shuffle))
    arguments.append(str(cluster))
    arguments.append(str(GPU))
    arguments.append(str(n_retrain))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            string_to_exec = "bsub python3 SingleRetraining.py "
        else:
            string_to_exec = "python3 SingleRetraining.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        print(string_to_exec)
        os.system(string_to_exec)
    i = i + 1
