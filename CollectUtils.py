import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def select_over_retrainings(folder_path, selection="error_train", mode="min", exact_solution=None):
    retrain_models = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    models_list = list()
    for retraining in retrain_models:
        # print("Looking for ", retraining)
        retrain_path = folder_path + "/" + retraining
        number_of_ret = retraining.split("_")[-1]
        # print("Looking for ", retrain_path + "/InfoModel.txt")
        if os.path.isfile(retrain_path + "/InfoModel.txt"):

            models = pd.read_csv(retrain_path + "/InfoModel.txt", header=None, sep=",", index_col=0)
            models = models.transpose()
            models["metric"] = models["loss_pde_no_norm"] + models["loss_vars"]
            models["retraining"] = number_of_ret

            models_list.append(models)
        else:
            print("No File Found")

    retraining_prop = pd.concat(models_list, ignore_index=True)
    retraining_prop = retraining_prop.sort_values(selection)
    # print("#############################################")
    # print(retraining_prop)
    # print("#############################################")
    if mode == "min":
        # print("#############################################")
        # print(retraining_prop.iloc[0])
        # print("#############################################")
        return retraining_prop.iloc[0]
    if mode == "max":
        # print("#############################################")
        # print(retraining_prop.iloc[0])
        # print("#############################################")
        return retraining_prop.iloc[-1]
    else:
        retraining = retraining_prop["retraining"].iloc[0]
        # print("#############################################")
        # print(retraining_prop.mean())
        # print("#############################################")
        retraining_prop = retraining_prop.mean()
        retraining_prop["retraining"] = retraining
        return retraining_prop


def select_over_retrainings_dist(folder_path, selection="error_train", mode="min"):
    retrain_models = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    models_list = list()
    for retraining in retrain_models:
        # print("Looking for ", retraining)
        retrain_path = folder_path + "/" + retraining
        number_of_ret = retraining.split("_")[-1]

        if os.path.isfile(retrain_path + "/InfoModel.txt"):
            models = pd.read_csv(retrain_path + "/InfoModel.txt", header=None, sep=",", index_col=0)
            models = models.transpose()
            models["retraining"] = number_of_ret
            models_list.append(models)
        else:
            print("No File Found")

    retraining_prop = pd.concat(models_list, ignore_index=True)
    retraining_prop = retraining_prop.sort_values(selection)
    if mode == "min":
        return retraining_prop.iloc[0]
    elif mode == "max":
        return retraining_prop.iloc[-1]
    else:
        retraining = retraining_prop["retraining"].iloc[0]
        retraining_prop = retraining_prop.mean()
        retraining_prop["retraining"] = retraining
        return retraining_prop
