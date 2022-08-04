import os
import sys
import json
import random
import subprocess

random.seed(42)

print("Start Retrainings")

sampling_seed = int(sys.argv[1])
n_coll = int(sys.argv[2])
n_u = int(sys.argv[3])
n_int = int(sys.argv[4])
folder_path = sys.argv[5]
validation_size = float(sys.argv[6])
network_properties = json.loads(sys.argv[7])
shuffle = sys.argv[8]
cluster = sys.argv[9]
GPU = str(sys.argv[10])
n_retrain = int(sys.argv[11])

seeds = list()
for i in range(n_retrain):
    seeds.append(random.randint(1, 500))
print(seeds)
import numpy as np
seeds = np.array(seeds)
seeds = (seeds - min(seeds))*49/(max(seeds) -min(seeds))
seeds = np.round(seeds).astype(np.int)

os.mkdir(folder_path)

for retrain in range(len(seeds)):
    folder_path_retraining = folder_path + "/Retrain_" + str(retrain)
    arguments = list()
    arguments.append(str(sampling_seed))
    arguments.append(str(n_coll))
    arguments.append(str(n_u))
    arguments.append(str(n_int))
    arguments.append(str(folder_path_retraining))
    arguments.append(str(validation_size))
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        arguments.append("\'" + str(network_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append(str(network_properties).replace("\'", "\""))
    arguments.append(str(seeds[retrain]))
    arguments.append(shuffle)

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            if GPU != "None":
                string_to_exec = "bsub -W 12:00 -R \"rusage[mem=16384,ngpus_excl_p=1]\" -R \"select[gpu_model0==" + GPU + "]\" python3 wPINNS.py  "
                print(string_to_exec)
            else:
                string_to_exec = "bsub -W 12:00 -R \"rusage[mem=8192]\" python3 wPINNS.py  "
        else:
            string_to_exec = "python3 AdversarialPINNS.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        os.system(string_to_exec)
    else:
        python = os.environ['PYTHON36']
        p = subprocess.Popen([python, "AdversarialPINNS.py"] + arguments)
        p.wait()
