import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import wandb
from algorithm import *
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


import sys
import time

def get_sweep_id(sweep_config,name,project, entity):
    """for parallel runs from .sbatch: provide 0 (start new sweep and save sweep ID) or i>0 to load previous sweep ID"""
    if len(sys.argv) > 1:
        parallel_number = int(sys.argv[1])  # Convert the argument to an integer
        print(f"Received argument parallel_number = {parallel_number }")
    else:
        parallel_number=0
        print("No argument provided from command line / sbatch.")
        # run with parallel_number =0? -> for normal runs, also without sbatch

    if parallel_number==0:
        sweep_id=wandb.sweep(sweep_config, project=project, entity=entity)
        with open(f"{name}.txt", "w") as file:
            file.write(sweep_id)
    else:
        time.sleep(15) #make sure process 0 is started already and "sweep_id_sweep_Adamff_exp2.txt" is saved
        with open(f"{name}.txt", "r") as file:
            sweep_id = file.read()

    return sweep_id, parallel_number



def init_run(size_string,i, project, pre_path_results, entity):
    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    identifier = f"{size_string}_{date_str}" 

    run = wandb.init(project=project,name = f'{identifier}_{i}' ,dir=pre_path_results, entity=entity)
    return run,identifier,date_str


def process_run(settings,run,data_module,pre_path,project,identifier,data_set_name,i,date_str,folder_name,str1=None,str2=None,shuffle=False):
    # folder_name should be e.g. student or teacher
    print('in process run')
    model_inst=get_inst(i,padding=3)

    neuron_synapse=f'{settings.neuron_name}_{settings.synapse.__name__}'
    if str1 is not None:
        identifier+='_'+str1
    if str2 is not None:
        identifier+='-'+str2
    if pre_path !='':
        pre_path+='/'
    model_path=f"{pre_path}results/{data_set_name}/{folder_name}/{neuron_synapse}/{identifier}/{model_inst}"
    os.makedirs(model_path, exist_ok = True) 

    run.log_model(path=model_path, name=f'{identifier}_{model_inst}')

    print(f'trying to log to model path: {model_path}')
    if pre_path !='':
                pre_path+='/'

    logger = WandbLogger(project=project,save_dir =f"{pre_path}results",
        name=f"{data_set_name}/{folder_name}/{neuron_synapse}/{identifier}/",
        version=f"{model_inst}",log_model="all")
    
    #redefine model with overwritten settings
    model = Lightning_SNN(
        net_size=settings.net_size,
        neuron_fct=settings.neuron_fct,
        neuron_intrinsic=settings.neuron_intrinsic,
        synapse_intrinsic=settings.synapse_intrinsic,
        train_intrinsic=settings.train_intrinsic,
        clip_intrinsic=settings.clip_intrinsic,
        regularizers=settings.regularizers,
        train_out=settings.train_out,
        train_hidden=settings.train_hidden,
        weight_info=settings.weight_info,
        nr=i,
        learning_rate=settings.learning_rate,
        time_step=settings.time_step,
        optim_class=settings.optim_class,
        optim_params=settings.optim_params,
        scheduler=settings.scheduler,
        surr_grad_scale=settings.surr_grad_scale,
        adapt_intrinsic=False,#TODO
        tau_mem_LIF=settings.tau_mem,
        training_mode = settings.training_mode,
        hidden_mode=settings.hidden_mode,
        target_model = settings.target_model,
        params_from_target = settings.params_from_target,
        hidden_in_loss = settings.hidden_in_loss,
        hidden_loss_scale=settings.hidden_loss_scale,
        noise=settings.noise,
        path_info=[pre_path, data_set_name, folder_name,identifier],
        N_train_only_weights = settings.N_train_only_weights,
        shuffle=shuffle
    )

    print('past model init')
    wandb.watch(model, log='all', log_freq=40)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{model_path}/checkpoints", every_n_epochs=1, save_top_k=-1,filename='{epoch}-{val_acc:.2f}',save_on_train_epoch_end=False
    )  

    print(f'logging checkpoints to {model_path}')

    if torch.cuda.is_available():
        print('\n\n cuda available\n\n')
        accelerator = 'auto'
    else:
        #lightning detects mps but should use GPU in that case
        accelerator = 'cpu'

    trainer = pl.Trainer(
                logger=logger,
                max_epochs=settings.N_epochs,
                accelerator=accelerator,
                #profiler=profiler,
                log_every_n_steps=1,#higher than actual steps per epoch -> only log at epoch end?
                limit_test_batches=5,  # only one test batch for plots
                check_val_every_n_epoch=1,
                #num_sanity_val_steps=1, #one validation step before training
                #log_every_n_steps = 1,
                callbacks=[checkpoint_callback,],
            )
    wandb.require(experiment="service")
    trainer.fit(model=model, train_dataloaders=data_module)  
    #test after training
    trainer.test(model,datamodule=data_module) 
    wandb.finish()

    
    os.system("wandb artifact cache cleanup 100MB")
    # comment out to save memory
    #os.system("rm -r /home/.cache/wandb/logs")
    #os.system("rm -r wandb")
    


def set_variables(data_set_name,config_path,seed, pre_path):
    torch.set_float32_matmul_precision('medium')
    identifier = None  
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed()

    settings = Config(data_set_name,settings_file=config_path)
    size_string=get_size_str(settings.batch_size,settings.nb_inputs,settings.nb_outputs)
    trainloader, valloader,testloader, nb_steps = choose_data_params(
        data_set_name, settings, num_workers=4,pre_path=pre_path
    ) 

    data_module = DataModule(trainloader, valloader, testloader)
    return data_module,size_string