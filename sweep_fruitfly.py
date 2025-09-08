import os
from algorithm import *
from helpers import *

pre_path_results = '/scratch/mie8014/results/SNNs/summerschool2025'


def run_agent(sweep_id, project_name):
    wandb.agent(sweep_id=sweep_id, function=train_model, project=project_name)


def train_model():

    pre_path_results = '/scratch/mie8014/results/SNNs/summerschool2025'
    project_name = "CC_SNN"
    entity = None  # add_your_entity

    run, identifier, date_str = init_run(
        size_string, seed, project_name, pre_path_results, entity
    )

    # ___________________
    # overwrite settings with wandb config from sweep and exclude run settinsg that are not needed
    if wandb.config.train_loss == "nll":
        train_layer = ""
    else:
        train_layer = f"_{wandb.config.train_layer}"

    settings = Config(wandb.config.data_set_name, settings_file="fruitfly")

    settings.hidden_in_loss = float(wandb.config.hil)
    settings.learning_rate = wandb.config.lr
    settings.surr_grad_scale = wandb.config.SG_scale
    settings.batch_size = wandb.config.batch
    settings.synapse_noise = wandb.config.synapse_noise
    settings.weight_scaling_factor = wandb.config.weight_scaling_factor
    settings.low_rank_sparse = wandb.config.low_rank_sparse
    settings.neuron_name = wandb.config.neuron_model
    settings.N_epochs = wandb.config.epochs
    settings.Dales_law = wandb.config.Dales_law
    settings.hidden_mode = wandb.config.hidden_mode
    settings.shuffle = wandb.config.shuffle

    if "BLK" in settings.neuron_name:
        settings.learning_rate = 0.01
    else:
        settings.learning_rate = wandb.config.lr

    if wandb.config.train_loss == "nll":
        # nll is not dependant on teacher's layer - no information about output layer added
        # information about hidden layer can be added when hil>0.0
        if wandb.config.train_layer == "out":
            wandb.finish()
            return
        elif wandb.config.train_layer == "hil":
            if float(wandb.config.hil) > 0.0:
                settings.train_hidden = True
                settings.hidden_mode = "mse"
                settings.hidden_in_loss = float(wandb.config.hil)
        else:
            raise (NotImplementedError)

    if wandb.config.train_layer == "out":
        # output layer gets no levels from hidden layer
        if float(wandb.config.hil) > 0.0:
            wandb.finish()
            return

    if wandb.config.weight_init == "uniform":
        settings.weight_init_ = almost_xavier_uniform_
    elif wandb.config.weight_init == "normal":
        settings.weight_init_ = almost_xavier_normal_

    if not wandb.config.range:
        settings.set_I_c_range = False
        settings.set_tau_mem_range = False
        settings.I_c_abs = 1.0  # =max initialized interval
        settings.tau_mem = 0.2  # > initialized interval

    settings.set_general_parameters()  # overwrite dependent parameters

    # create dataset with new settings (e.g. batch size)
    trainloader, valloader, testloader, nb_steps = choose_data_params(
        wandb.config.data_set_name, settings, num_workers=16, pre_path=pre_path_data
    )
    data_module = DataModule(trainloader, valloader, testloader)
    print("past all init")
    # ___________________

    str1 = f"{wandb.config.train_params}_{wandb.config.weight_init}_{wandb.config.i}"
    if wandb.config.low_rank_sparse is None:
        str2 = f"sparse{int(wandb.config.sparsity*10)}"
    elif wandb.config.low_rank_sparse == 1.0:
        str2 = "lowrank"

    folder_name = f"student/{wandb.config.train_params}"

    process_run(
        settings,
        run,
        data_module,
        pre_path_results,
        project_name,
        identifier,
        data_set_name,
        wandb.config.j,
        date_str,
        folder_name=folder_name,
        str1=str1,
        str2=str2,
        shuffle=wandb.config.shuffle,
    )


if __name__ == "__main__":

    config_path = "fruitfly"  # only used for data and network size
    project_name = "CC_SNN"
    data_set_name = "olfactory"  #'rms','rml','shd'
    pre_path_data = "/scratch/mie8014/data/SNNs/summerschool2025/samples2"#"data/ORN_data/samples"
    entity = None  # add_your_entity

    N_teachers = 1
    N_students_per_teacher = 3

    sweep_config = {
        "method": "grid",
        "name": f"{data_set_name}:nont_val",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "data_set_name": {"values": [data_set_name]},
            "N_teachers": {"values": [N_teachers]},  # repetitions
            "i": {"values": list(range(N_teachers))},  # repetition indices
            "j": {"values": list(range(N_students_per_teacher))},  # repetition indices
            "hil": {"values": [0.0]},
            "epochs": {
                "values": [
                    15,
                ]
            },
            "range": {
                "values": [1]
            },  # time constant and baseline current are initialized heterogeneously
            "hidden_mode": {
                "values": [
                    "mse",
                ]
            },
            "SG_scale": {"values": [50]},#{"values": [30, 50, 100]},
            "lr": {"values": [0.01]},#{"values": [0.001, 0.01, 0.1]},
            "batch": {"values": [128]},
            "weight_scaling_factor": {"values": [0.2]},
            "teacher_weight_init": {"values": ["normal"]},  # ['uniform','normal']},
            "teacher_low_rank_sparse": {"values": [None]},  # [None,1.0]},
            "teacher_sparsity": {"values": [0.5]},  # ,0.8,0.5,0.2]},
            "train_params": {"values": ["oi"]},
            "train_layer": {"values": ["hil"]},
            "train_loss": {"values": ["nll"]},
            "low_rank_sparse": {"values": [None]},  # [None,1.0]},
            "sparsity": {
                "values": [1.0]
            },  # WARNING - when changing this value, also change conditions above!!! #otherwise, teacher's pattern cannot be learned
            "weight_init": {"values": ["normal"]},  # ['uniform','normal']},
            "neuron_model": {"values": ["LIF"]},
            "synapse_noise": {"values": [0.00, 0.02]},
            "teacher_neuron_model": {"values": ["LIF"]},
            "Dales_law": {"values": [None]},
            "shuffle": {
                "values": [0]
            },  # for 1:different baseline where student is initialized as shuffled teacher but then trained
        },
    }

    sweep_id, seed = get_sweep_id(
        sweep_config, f"students_{data_set_name}", project_name, entity
    )

    data_module, size_string = set_variables(
        data_set_name, config_path, seed, pre_path_data
    )

    run_agent(sweep_id, project_name)
