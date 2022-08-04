from CollectUtils import *

np.random.seed(42)
file_ex = "Data/BurgersExact.txt"
exact_solution = np.loadtxt(file_ex)

base_path_list = ["RarefactionWave"]

for base_path in base_path_list:
    print("#################################################")
    print(base_path)

    b = False
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sensitivity_df = pd.DataFrame()
    selection_criterion = "metric"
    eval_metric = "rel_L2_norm"
    threshold = 0.005
    plot_color = "reset-freq"
    mode = "min"
    mode_ret = "mean"

    Nu_list = []
    Nf_list = []

    L2_norm = []
    criterion = []
    best_retrain_list = []
    list_models_setup = list()

    for subdirec in directories_model:
        print(subdirec)
        model_path = base_path

        sample_path = model_path + "/" + subdirec
        retrainings_fold = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]

        retr_to_check_file = None
        for ret in retrainings_fold:
            print(sample_path + "/" + ret + "/EnsembleInfo.csv")
            if os.path.isfile(sample_path + "/" + ret + "/EnsembleInfo.csv"):
                retr_to_check_file = ret
                break

        setup_num = int(subdirec.split("_")[1])
        if retr_to_check_file is not None:
            info_model = pd.read_csv(sample_path + "/" + retr_to_check_file + "/EnsembleInfo.csv", header=None, sep=",", index_col=0)
            info_model = info_model.transpose().reset_index().drop("index", 1)
            best_retrain = select_over_retrainings(sample_path, selection=selection_criterion, mode=mode_ret, exact_solution=exact_solution)
            best_retrain = best_retrain.to_frame()
            best_retrain = best_retrain.transpose().reset_index().drop("index", 1)
            info_model = pd.concat([info_model, best_retrain], 1)
            info_model["setup"] = setup_num
            sensitivity_df = sensitivity_df.append(info_model, ignore_index=True)
        else:
            print(sample_path + "/Information.csv not found")

    sensitivity_df = sensitivity_df.sort_values(selection_criterion)

    if mode == "min":
        best_setup = sensitivity_df.iloc[0]
    elif mode == "max":
        best_setup = sensitivity_df.iloc[-1]
    else:
        raise ValueError()
    # print(sensitivity_df)
    print("Best Setup:", best_setup["setup"])
    print(best_setup)
    best_setup.to_csv(base_path + "/best.csv", header=0, index=True)

    sensitivity_df = sensitivity_df.rename(columns={'reset_freq': 'reset-freq'})

    plt.figure()
    plt.grid(True, which="both", ls=":")
    sns.scatterplot(data=sensitivity_df, x=selection_criterion, y=eval_metric, hue=plot_color)

    plt.xlabel(r'$\varepsilon_T$')
    plt.ylabel(r'$\varepsilon$')
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(base_path + "/et_vs_eg_" + selection_criterion + "_" + mode_ret + ".png", dpi=400)
