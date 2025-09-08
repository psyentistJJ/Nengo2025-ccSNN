import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import wandb
import matplotlib.colors as clr
import seaborn as sns
import copy
from sklearn.metrics import confusion_matrix

import torch
import pytorch_lightning as pl


import gc

from algorithm.model.str_helpers import *
from algorithm.model.fruitfly_olf import SNN
from algorithm.neuron.plotting_helpers import plot_voltage_traces
from algorithm.metrics import *


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it -> from https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    img = [wandb.Image(img)]  # added line
    return img


'''
class weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        """weighted MSE, weight should be of dimensionality (timesteps, neurons)"""
        batch_mean = np.mean((input - target)**2, dim=0)
        print(f'batch mean shape is {batch_mean.shape}')
        return np.mean(batch_mean * weight)'''

"""elif self.hidden_mode=='mse_weighted':
                    if self.before_first_epoch:
                        self.loss_fn = weighted_MSELoss()"""


class Lightning_SNN(pl.LightningModule):
    def __init__(
        self,
        net_size,
        neuron_fct,
        neuron_intrinsic,
        synapse_intrinsic,
        time_step,
        learning_rate,
        weight_info,
        optim_class,
        optim_params=(0.9, 0.999),
        scheduler=None,
        regularizers=None,
        train_out=True,
        train_hidden=False,
        train_intrinsic=[],
        clip_intrinsic={},
        nr=0,
        surr_grad_scale=100.0,
        adapt_intrinsic=False,
        tau_mem_LIF=None,
        training_mode="classification",
        hidden_mode="mse",
        target_model=None,
        params_from_target=None,
        hidden_in_loss=None,
        hidden_loss_scale=1,
        noise=None,
        path_info=None,
        shuffle=False,
        N_train_only_weights=0,
    ):
        """
        Class for spiking neural network with inout, hidden and output layer.

        arguments
            net_size: list of int
                batch_size: batch size
                nb_inputs: number of input neurons
                nb_hidden: number of hidden neurons
                nb_outputs: number of output neurons
            neuron_fct: function
                describes neuron model, choose from {LIF, LIF2, QIF}
            neuron_intrinsic: list of float or list of np.ndarray
                U_L: leak potential
                tau_mem: membrane time constant
                thresh: spiking threshold voltage
                reset: reset voltage
                a: scaling parameter for QIF dv/dy
                I_c: current that determines spike onset for QIF
            regularizers: TODO: reg classes
                None, None or lower and upper activity regularizers
            train_weights: bool
                whether to train network weights or not
            train_intrinsic: list of str
                list of neuron_intrsinsic attributes that should be trained
            clip_intrinsic: list
                intervals to clip trained parameters to
            nb_steps: int
                (max) number of steps per forward pass
        """
        self.call_outside_loop = False

        super().__init__()
        self.save_hyperparameters(ignore=["target_model"])

        print(f"lightning init device {self.device}")

        self.inst = get_inst(nr, padding=3)

        self.target_model = target_model
        if self.target_model is not None:
            self.target_model = self.target_model.eval()
            print(f"target v in SNN: {self.target_model.v}")
            model_is_student = True
            self.target_shuffled = None
        else:
            model_is_student = False

        weight_info = self.check_weight_init(weight_info)

        params_from_target = self.check_param_init(params_from_target)

        self.model = SNN(
            net_size=net_size,
            neuron_fct=neuron_fct,
            neuron_intrinsic=neuron_intrinsic,
            synapse_intrinsic=synapse_intrinsic,
            regularizers=regularizers,
            train_intrinsic=train_intrinsic,
            clip_intrinsic=clip_intrinsic,
            weight_info=weight_info,
            time_step=time_step,
            surr_grad_scale=surr_grad_scale,
            adapt_intrinsic=adapt_intrinsic,
            tau_mem_LIF=tau_mem_LIF,
            training_mode=training_mode,
            params_from_target=params_from_target,
            hidden_in_loss=hidden_in_loss,
            hidden_loss_scale=hidden_loss_scale,
            noise=noise,
            is_student=model_is_student,
            shuffle=shuffle,
            N_train_only_weights=N_train_only_weights,
        )

        if params_from_target is not None:
            for name, value in self.model.named_parameters():
                if name in params_from_target:
                    print(f"in Lightning SNN: model {name} is {value}")

        self.batch_size = self.model.batch_size

        # copy from inputs
        self.learning_rate = learning_rate
        self.optim_class = optim_class
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.train_hidden = train_hidden
        self.train_out = train_out
        self.hidden_mode = hidden_mode
        print(
            f"train out: {self.train_out}, train hidden: {self.train_hidden}, self hidden mode is {self.hidden_mode}"
        )

        # logic to save metrics
        self.on_step_b = False
        self.on_epoch_b = True
        self.logger_b = True
        self.prog_bar_b = True

        self.batch_padded = None

        self.model_path = self.set_path(path_info)
        self.save_init_weights()

        self.before_first_epoch = True
        self.before_first_val = True

        self.switch_intr_weights = False

        self.all_preds = []
        self.all_preds_teacher = []
        self.all_targets = []

        # initial test step??

    def set_path(self, path_info):
        if path_info is not None:
            [pre_path, data_set_name, folder_name, identifier] = path_info
            if (
                identifier is None
            ):  # use size & date if no other identifier string provided
                identifier = f"{self.model.size_string}_{date_str}"
            if pre_path != "":
                pre_path += "/"
            model_path = f"{pre_path}results/{data_set_name}/{folder_name}/{self.model.neuron_name}_{self.model.synapse_name}/{identifier}/{self.inst}"
        else:
            model_path = None
        return model_path

    def check_weight_init(self, weight_info):

        (
            train_weights,
            Dales_law,
            weight_init_,
            target_weights,
            weight_scaling_factor,
            recurrent_sparse,
            low_rank_sparse,
        ) = weight_info

        if target_weights is not None and target_weights == "target_model":
            weight_init_dict = {}
            for name, value in self.target_model.named_parameters():
                weights = ["w1", "w2", "v", "init_v"]
                if low_rank_sparse is not None:
                    weights += ["subv1", "subv2"]
                print(f"weight {name} is value {value.shape}")
                if name in weights:
                    if "v" in name and (not "subv1" in name or "subv2" in name):
                        try:  # for runs with only intrinsic trained, init_v is saved instead of v
                            print(f"weight {name}")
                            if value is not None:
                                weight_init_dict["v"] = value
                        except:
                            pass
                    else:
                        print(f"weight {name}")
                        weight_init_dict[name] = value
                    # print(f'weight {name} is value {value}')
            print("weight init from target model")
        else:
            weight_init_dict = None

        weight_info = (
            train_weights,
            Dales_law,
            weight_init_,
            weight_init_dict,
            weight_scaling_factor,
            recurrent_sparse,
            low_rank_sparse,
        )
        return weight_info

    def set_split_train_inrt_weights(self):
        """check if current epoch is in last last N epochs, if so, freeze intrinsic parameters and instead train some of the weights"""
        if (not self.switch_intr_weights) and (
            self.trainer.max_epochs - self.current_epoch
        ) <= self.model.N_train_only_weights:
            for name, value in self.model.named_parameters():
                if (not value.requires_grad) and (
                    name in self.model.train_weights_later
                ):
                    value.requires_grad = True
                elif (
                    value.requires_grad
                ):  # and name in self.model.train_intrinsic: fix all previously trained params
                    value.requires_grad = False
            self.optimizers = self.configure_optimizers()
            self.switch_intr_weights = True  # only switch once

    def save_init_weights(self):
        if self.model_path is not None:
            os.makedirs(self.model_path, exist_ok=True)
            torch.save(self.model.w1, f"{self.model_path}/init_w1")
            torch.save(self.model.v, f"{self.model_path}/init_v")
            torch.save(self.model.w2, f"{self.model_path}/init_w2")
            if "beta" in self.model.train_intrinsic:
                torch.save(self.model.beta, f"{self.model_path}/init_beta")
            if "I_c" in self.model.train_intrinsic:
                torch.save(self.model.I_c, f"{self.model_path}/init_I_c")
            if "reset" in self.model.train_intrinsic:
                torch.save(self.model.reset, f"{self.model_path}/init_reset")

    def check_param_init(self, params_from_target):
        params = None
        if params_from_target is not None:
            params = {}
            for name, value in self.target_model.named_parameters():
                if name in params_from_target:
                    params[name] = value
                    print(f"in Lightning SNN: target {name} is {value}")
            # for name in params_from_target:
            #   params[name] = self.target_model.__getattribute__(name)
        return params

    def forward(self, inputs, target):
        """ """
        # print(f"device in lightning forward: {self.device}")
        # print(f"batch_size in lightning forward: {self.batch_size}")
        self.set_split_train_inrt_weights()  # potentially freeze intrinsic and only train some of weights
        out_rec, other_recs = self.model(inputs, target)
        self.batch_padded = self.model.batch_padded
        return out_rec, other_recs

    def target_forward(self, inputs, target):
        """ """
        # print(f"device in lightning forward: {self.device}")
        # print(f"batch_size in lightning forward: {self.batch_size}")
        print(f"in forward: {self.target_model.v}")
        out_rec, other_recs = self.target_model(inputs, target)
        # self.batch_padded = self.model.batch_padded
        return out_rec, other_recs

    def target_forward_shuffled(self, inputs, target):
        if self.target_shuffled is None:
            self.target_shuffled = copy.deepcopy(self.target_model).eval()
            self.target_shuffled.low_rank_sparse = None  # make sure that shuffled v is used in fprward pass instead of product of subv
        shuffle_idx = np.arange(self.model.nb_hidden)
        np.random.shuffle(shuffle_idx)
        self.target_shuffled.w1.data = self.target_shuffled.w1.data[:, shuffle_idx]
        self.target_shuffled.w2.data = self.target_shuffled.w2.data[shuffle_idx, :]
        self.target_shuffled.v.data = self.target_shuffled.v.data[shuffle_idx, :]
        self.target_shuffled.v.data = self.target_shuffled.v.data[:, shuffle_idx]

        out_rec, other_recs = self.target_shuffled.forward(inputs, target)

        return out_rec, other_recs

    def configure_optimizers(self):
        #params_no_decay = self.model.train_intrinsic

        params_small_lr = [p for name, p in self.model.named_parameters() if 'nt_values' in name]
        others = [p for name, p in self.model.named_parameters() if 'nt_values' not in name]

        #very small learning rate for weight scaling per neurotransmitter -> finetuning only
        grouped_parameters =  [{'params': params_small_lr, 'lr': 1e-6},{'params': others}]

        # set regularization ONLY for weight parameters, not for neuron-intrinsic (they get clipped anyway)
        # now weight decay for now
        '''grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in params_no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in params_no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]'''
        print(f'optimizer params: {grouped_parameters}')
        optimizer = self.optim_class(
            grouped_parameters, self.learning_rate, self.optim_params
        )
        # optimizer = self.optim_class(
        #    self.model.parameters(), self.learning_rate,self.optim_params)#betas=(0.9, 0.999) for adam and adaax, momentum for SGD
        if self.scheduler is not None:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.scheduler["base_lr"],
                max_lr=self.scheduler["max_lr"],
                step_size_up=self.scheduler["step_size_up"],
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            return optimizer
        """
        optimizer = self.optim_class(
            self.model.parameters(), self.learning_rate,self.optim_params)#betas=(0.9, 0.999) for adam and adaax, momentum for SGD
        if self.scheduler is not None:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.scheduler['base_lr'], max_lr=self.scheduler['max_lr'],step_size_up=self.scheduler['step_size_up'])
            return [optimizer], [{'scheduler': scheduler, 'interval': "epoch"}]
        else:
            return optimizer
            """

    def get_loss_out(self, model_output, target, N):
        if (
            self.model.training_mode == "classification"
            or self.model.training_mode == "classification_BLK"
        ):
            log_p_y = self.model.log_softmax_fn(
                model_output
            )  # here, m (max over time and units) is used as model_output
            print(f"log_p_y is shape {log_p_y.shape},  {log_p_y}")
            print(f'output min: {torch.min(model_output[0,:],0).values}')
            print(f'output max: {torch.max(model_output[0,:],0).values}')
            loss = self.model.loss_fn(log_p_y[:N], target.to(torch.int64))
        elif (
            self.model.training_mode == "trace_learning"
            or self.model.training_mode == "trace_learning_BLK"
        ):
            # print(f'\n\nmodel output {model_output.shape}, :N: {model_output[:N].shape}\n\n') #here, full traces of all output neurons of student and teacher are used
            loss = self.model.loss_fn(model_output[:N], target[:N])

        return loss

    def get_regularization(self, output):
        if self.model.regularize:
            L_reg = (
                1
                / self.batch_size
                * (
                    self.model.reg_min_activity(output)
                    + self.model.reg_max_activity(output)
                )
            )  # Zenke & Vogels (2021)
            print(f"L_reg is {L_reg}")
        else:
            L_reg = 0.0
        return L_reg

    def get_acc(self, m, target, N):
        _, am = torch.max(m, 1)  # argmax over output units
        acc = np.mean((target == am[:N]).detach().cpu().numpy())  # compare to labels
        return acc

    def get_N_spikes(self, spk_rec, target):
        M = len(target)  # nr of hidden neurons
        spikes_per_neuron = spk_rec[:M, :, :].sum(
            axis=1
        )  # sum over time, :M -> in case of padding, use only 'real' hidden layer activities
        av_spikes_per_neuron = spikes_per_neuron.mean()
        print(f'spikes per neuron mean: {av_spikes_per_neuron}')
        perc_neuron_spiking = (spikes_per_neuron > 0).sum() / (M * self.model.nb_hidden)

        return av_spikes_per_neuron, perc_neuron_spiking

    def get_MSE(self, name=None, pred=None, target=None, mode="all", shuffle=False):
        """MSE between voltage traces or parameters of neurons"""
        if name is None:
            # print(f'pred: {pred.shape}, target: {target.shape}')
            MSE = torch.mean((pred - target) ** 2)
        else:
            # return MSE for all or parts of hidden units
            if mode == "all":
                mask = self.model.mask_all
            elif mode == "rec":
                mask = self.model.mask_rec
            elif mode == "unrec":
                mask = self.model.mask_unrec
            else:
                raise NotImplementedError
            if shuffle:
                idx = torch.randperm(self.target_model.state_dict()[name].size(0))
                prediction = self.target_model.state_dict()[name][idx][mask]
            else:
                prediction = self.model.state_dict()[name][mask]
            target = self.target_model.state_dict()[name][mask]
            MSE = torch.mean((prediction - target) ** 2)
        return MSE

    def get_STP(self, spk_rec, target_spk_rec, mode="all"):
        # all_spikes_rec = torch.flatten(spk_rec) #attach all spikes for all neurons together (??)
        # all_spikes_target = torch.flatten(target_spk_rec) #attach all spikes for all neurons together (??)

        if mode == "all":
            mask = self.model.mask_all
        elif mode == "rec":
            mask = self.model.mask_rec
        elif mode == "unrec":
            mask = self.model.mask_unrec
        else:
            raise NotImplementedError

        Nbatch, Nt, Nhidden = spk_rec.shape
        STP = spike_time_precision(
            spk_rec[:, :, mask],
            target_spk_rec[:, :, mask],
            Δ=self.model.time_step,
            duration=Nt * Nhidden,
            dt=self.model.time_step,
        )
        return STP

    def get_metrics_classification(self, m, target, spk_rec, mode):
        N = target.size()[0]
        # for padded batches -> use only unpadded parts for loss/acc

        if self.model.is_spiking:
            # regularization (upper/lower bound for spikes)
            reg = self.get_regularization(spk_rec)  # 0 if no regularization
        else:
            # regularize to smaller voltage values??
            reg = 10 * torch.mean(spk_rec)

        # get loss four output layer
        loss_out = self.get_loss_out(m, target, N)

        acc = self.get_acc(m, target, N)

        values = {
            f"{mode}_loss_out": loss_out,
            f"{mode}_spk_reg": reg,
            f"{mode}_acc": acc,
        }  # add more items if needed

        if self.model.is_spiking:
            av_spikes, perc_spiking = self.get_N_spikes(spk_rec, target)
            values[f"{mode}_av_spikes_per_neuron"] = av_spikes
            values[f"{mode}_perc_neurons_spiking"] = perc_spiking

        if not self.call_outside_loop:
            self.log_dict(
                values,
                on_step=self.on_step_b,
                on_epoch=self.on_epoch_b,
                prog_bar=self.prog_bar_b,
                logger=self.logger_b,
                batch_size=self.batch_size,
            )

        return loss_out, reg

    def get_metrics_trace_learning(
        self,
        output,
        target_output,
        other_recordings,
        target_other_recordings,
        target,
        mode,
    ):
        N = target.size()[0]
        m, _ = torch.max(output, 1)
        mem_rec, spk_rec = other_recordings
        # for padded batches -> use only unpadded parts for loss/acc
        reg = self.get_regularization(spk_rec)  # 0 if no regularization
        loss_out = self.get_loss_out(output, target_output, N)
        acc = self.get_acc(m, target, N)

        values = {
            f"{mode}_loss_out": loss_out,
            f"{mode}_reg": reg,
            f"{mode}_acc": acc,
        }  # add more items if needed

        if self.model.is_spiking:
            av_spikes, perc_spiking = self.get_N_spikes(spk_rec, target)
            values[f"{mode}_av_spikes_per_neuron"] = av_spikes
            values[f"{mode}_perc_neurons_spiking"] = perc_spiking

        self.log_dict(
            values,
            on_step=self.on_step_b,
            on_epoch=self.on_epoch_b,
            prog_bar=self.prog_bar_b,
            logger=self.logger_b,
            batch_size=self.batch_size,
        )
        return loss_out, reg

    def get_additional_metrics(
        self,
        model,
        output,
        target_output,
        other_recordings,
        target_other_recordings,
        mode,
    ):

        hidden_loss = 0

        values = {}
        mem_rec, spk_rec = other_recordings
        target_mem_rec, target_spk_rec = target_other_recordings

        if self.before_first_epoch:
            print(f"mem_rec shape: {mem_rec.shape}")
            print(f"spk_rec shape: {spk_rec.shape}")
            print(f"output shape: {output.shape}")
            print(f'w1: {self.model.w1.values()}')
            print(f'v: {self.model.v.values()}')

        # values[f'{mode}_ISID_hidden_all'],values[f'{mode}_SD_hidden_all'],values[f'{mode}_RISD_hidden_all'],values[f'{mode}_SS_hidden_all']=spk_metrics(spk_rec, target_spk_rec,self.model.time_step)
        # values[f'{mode}_sync_SPK_hidden_all']=sync_metrics(spk_rec, target_spk_rec,self.model.time_step)
        values[f"{mode}_corr_V_hidden_all"] = get_corr(mem_rec, target_mem_rec)
        values[f"{mode}_corr_SPK_hidden_all"] = get_corr(spk_rec, target_spk_rec)
        values[f"{mode}_corr_V_out_all"] = get_corr(output, target_output)
        values[f"{mode}_MSE_V_out_all"] = self.get_MSE(
            pred=output, target=target_output
        )
        values[f"{mode}_MSE_V_hidden_all"] = self.get_MSE(
            pred=mem_rec, target=target_mem_rec
        )
        values[f"{mode}_MSE_SPK_hidden_all"] = self.get_MSE(
            pred=spk_rec, target=target_spk_rec, mode="all"
        )

        # values[f'{mode}_ISID_hidden_unrec'],values[f'{mode}_SD_hidden_unrec'],values[f'{mode}_RISD_hidden_unrec'],values[f'{mode}_SS_hidden_unrec']=spk_metrics(spk_rec[:,:,self.model.mask_unrec], target_spk_rec[:,:,self.model.mask_unrec],self.model.time_step)
        # values[f'{mode}_sync_SPK_hidden_unrec']=sync_metrics(spk_rec[:,:,self.model.mask_unrec], target_spk_rec[:,:,self.model.mask_unrec],self.model.time_step)
        values[f"{mode}_corr_V_hidden_unrec"] = get_corr(
            mem_rec[:, :, self.model.mask_unrec],
            target_mem_rec[:, :, self.model.mask_unrec],
        )
        values[f"{mode}_corr_SPK_hidden_unrec"] = get_corr(
            spk_rec[:, :, self.model.mask_unrec],
            target_spk_rec[:, :, self.model.mask_unrec],
        )
        values[f"{mode}_MSE_V_hidden_unrec"] = self.get_MSE(
            pred=mem_rec[:, :, self.model.mask_unrec],
            target=target_mem_rec[:, :, self.model.mask_unrec],
        )
        values[f"{mode}_MSE_SPK_hidden_unrec"] = self.get_MSE(
            pred=spk_rec[:, :, self.model.mask_unrec],
            target=target_spk_rec[:, :, self.model.mask_unrec],
        )

        # values[f'{mode}_ISID_hidden_rec'],values[f'{mode}_SD_V_hidden_rec'],values[f'{mode}_RISD_V_hidden_rec'],values[f'{mode}_SS_hidden_rec']=spk_metrics(spk_rec[:,:,self.model.mask_rec], target_spk_rec[:,:,self.model.mask_rec],self.model.time_step)
        # values[f'{mode}_sync_SPK_hidden_rec']=sync_metrics(spk_rec[:,:,self.model.mask_rec], target_spk_rec[:,:,self.model.mask_rec],self.model.time_step)
        values[f"{mode}_corr_V_hidden_rec"] = get_corr(
            mem_rec[:, :, self.model.mask_rec],
            target_mem_rec[:, :, self.model.mask_rec],
        )
        values[f"{mode}_corr_SPK_hidden_rec"] = get_corr(
            spk_rec[:, :, self.model.mask_rec],
            target_spk_rec[:, :, self.model.mask_rec],
        )
        values[f"{mode}_MSE_V_hidden_rec"] = self.get_MSE(
            pred=mem_rec[:, :, self.model.mask_rec],
            target=target_mem_rec[:, :, self.model.mask_rec],
        )
        values[f"{mode}_MSE_SPK_hidden_rec"] = self.get_MSE(
            pred=spk_rec[:, :, self.model.mask_rec],
            target=target_spk_rec[:, :, self.model.mask_rec],
        )

        if self.model.is_spiking:
            # metrics that only work for spike train, not for RNN output
            values[f"{mode}_STP_V_hidden_all"] = self.get_STP(
                spk_rec, target_spk_rec, mode="all"
            )
            values[f"{mode}_STP_hidden_unrec"] = self.get_STP(
                spk_rec, target_spk_rec, mode="unrec"
            )
            values[f"{mode}_STP_hidden_rec"] = self.get_STP(
                spk_rec, target_spk_rec, mode="rec"
            )

            values[f"{mode}_sync_SPK_hidden_all"] = sync_metrics(
                spk_rec, target_spk_rec, self.model.time_step
            )
            values[f"{mode}_sync_SPK_hidden_unrec"] = sync_metrics(
                spk_rec[:, :, self.model.mask_unrec],
                target_spk_rec[:, :, self.model.mask_unrec],
                self.model.time_step,
            )
            values[f"{mode}_sync_SPK_hidden_rec"] = sync_metrics(
                spk_rec[:, :, self.model.mask_rec],
                target_spk_rec[:, :, self.model.mask_rec],
                self.model.time_step,
            )

        if "shuffled" in mode:
            # shuffled teacher instead of student
            shuffle = True
        else:
            # student
            shuffle = False
        for name in self.model.train_intrinsic:

            values[f"{mode}_MSE_{name}_hidden_all"] = self.get_MSE(
                name=name, mode="all", shuffle=shuffle
            )
            values[f"{mode}_MSE_{name}_hidden_rec"] = self.get_MSE(
                name=name, mode="rec", shuffle=shuffle
            )
            values[f"{mode}_MSE_{name}_hidden_unrec"] = self.get_MSE(
                name=name, mode="unrec", shuffle=shuffle
            )
        if self.model.training_mode == "trace_learning_BLK":
            v_rec = model.v[self.model.mask_rec, :][:, self.model.mask_rec]
            v_unrec = model.v[self.model.mask_unrec, :][:, self.model.mask_unrec]

            target_v_rec = self.target_model.v[self.model.mask_rec, :][
                :, self.model.mask_rec
            ]
            target_v_unrec = model.v[self.model.mask_unrec, :][:, self.model.mask_unrec]
            values[f"{mode}_MSE_v_all"] = self.get_MSE(
                pred=model.v, target=self.target_model.v
            )
            values[f"{mode}_MSE_v_rec"] = self.get_MSE(pred=v_rec, target=target_v_rec)
            values[f"{mode}_MSE_v_unrec"] = self.get_MSE(
                pred=v_unrec, target=target_v_unrec
            )

        if self.model.hidden_in_loss is not None:
            if self.hidden_mode == "mse":
                hidden_loss = values[
                    f"{mode}_MSE_V_hidden_rec"
                ]  # self.get_MSE(pred=output[self.model.hidden_in_loss], target=target_output[self.model.hidden_in_loss])

            elif self.hidden_mode == "bce":
                if self.before_first_epoch:
                    self.loss_fn = torch.nn.BCEWithLogitsLoss()
                hidden_loss = self.loss_fn(
                    spk_rec[:, :, self.model.mask_rec],
                    target_spk_rec[:, :, self.model.mask_rec],
                )
                print(f"hidden loss is {self.loss_fn}")
                values[f"{mode}_BCE_hidden_rec"] = hidden_loss
            else:
                raise NotImplementedError

            if self.model.hidden_loss_scale is not None:
                # scale here, before logging
                hidden_loss = self.model.hidden_loss_scale * hidden_loss
                values[f"{mode}_loss_hidden"] = hidden_loss

        if mode == "train" and self.before_first_epoch:
            self.plot_activities_student_teacher(
                spk_rec, target_spk_rec, mem_rec, target_mem_rec, output, target_output
            )
            self.before_first_val = False
        elif mode == "test":
            self.plot_activities_student_teacher(
                spk_rec, target_spk_rec, mem_rec, target_mem_rec, output, target_output
            )

        if not self.call_outside_loop:
            self.log_dict(
                values,
                on_step=self.on_step_b,
                on_epoch=self.on_epoch_b,
                prog_bar=self.prog_bar_b,
                logger=self.logger_b,
                batch_size=self.batch_size,
            )

        return hidden_loss

    def log_weights(self):
        values = {}
        for name, value in self.named_parameters():
            values[f"mean_{name}"] = value.mean()
            values[f"std_{name}"] = value.std()

        if not self.call_outside_loop:
            self.log_dict(
                values,
                on_step=self.on_step_b,
                on_epoch=self.on_epoch_b,
                prog_bar=self.prog_bar_b,
                logger=self.logger_b,
                batch_size=self.batch_size,
            )

    # def train_iters(self, inputs, target, iters):
    def training_step(self, batch, batch_idx):
        data, target, duration = batch

        print(f'sum input spikes: {torch.sum(data)}')

        if self.before_first_epoch:
            print(f'w1: {self.model.w1.values()}')
            print(f'v: {self.model.v.values()}')

        output, other_recordings = self.forward(data, target)

        if self.before_first_epoch:
            self.plot_intrinsic_distr()
            mem_rec, spk_rec = other_recordings
            self.plot_mem_hidden(mem_rec, spk_rec)
            self.plot_mem_out(output)
            self.plot_v()
            self.plot_reset()

        if (
            self.target_model is not None
        ):  # evaluate batch with target mdoel for metrics, even if NLL classification used
            target_output, target_other_recordings = self.target_forward(data, target)

        if (
            self.model.training_mode == "classification"
            or self.model.training_mode == "classification_BLK"
        ):
            mem_rec, spk_rec = other_recordings
            m, _ = torch.max(output, 1)
            loss_out, reg = self.get_metrics_classification(m, target, spk_rec, "train")
        elif (
            self.model.training_mode == "trace_learning"
            or self.model.training_mode == "trace_learning_BLK"
        ):
            loss_out, reg = self.get_metrics_trace_learning(
                output,
                target_output,
                other_recordings,
                target_other_recordings,
                target,
                "train",
            )

        loss_hidden = 0
        if self.target_model is not None:
            loss_hidden = self.get_additional_metrics(
                self.model,
                output,
                target_output,
                other_recordings,
                target_other_recordings,
                "train",
            )

        # add l2 norm for classifier
        w2_reg = 0.000001 * torch.norm(self.model.w2, p=1)
        values={}#dict for logging
        values["train_w2_reg"] = w2_reg
        reg += w2_reg

        loss = reg
        if self.train_hidden:
            loss += loss_hidden
        if self.train_out:
            loss += loss_out

        values["train_loss"] = loss
        values["train_w2_mean"] = torch.mean(self.model.w2)
        values["train_w2_std"] = torch.std(self.model.w2)

        if self.scheduler is not None:
            values["train_lr_scheduled"] = self.lr_schedulers().get_last_lr()[0]

        if not self.call_outside_loop:
            self.log_dict(
                values,
                on_step=self.on_step_b,
                on_epoch=self.on_epoch_b,
                prog_bar=self.prog_bar_b,
                logger=self.logger_b,
                batch_size=self.batch_size,
            )

        self.before_first_epoch = False
        return loss

    def on_after_backward(self):
        """after backward, BEFORE optimizer step: clip parameters and log gradients for all parameters"""
        self.model.clip_params(self.device)  # here instead of after whole epoch

        global_step = self.global_step

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param is not None, f"{name}"
                assert param.grad is not None, f"{name}"

    def validation_step(self, batch, batch_idx):
        data, target, duration = batch

        output, other_recordings = self.forward(data, target)

        if (
            self.target_model is not None
        ):  # evaluate batch with target model for metrics, even if NLL classification used
            target_output, target_other_recordings = self.target_forward(data, target)
            target_output_shuffled, target_other_recordings_shuffled = (
                self.target_forward_shuffled(data, target)
            )
            m_target_shuffled, _ = torch.max(target_output_shuffled, 1)

        if (
            self.model.training_mode == "classification"
            or self.model.training_mode == "classification_BLK"
        ):
            mem_rec, spk_rec = other_recordings
            m, _ = torch.max(output, 1)
            loss_out, reg = self.get_metrics_classification(m, target, spk_rec, "val")
            if self.target_model is not None:
                mem_rec_target_shuffled, spk_rec_target_shuffled = (
                    target_other_recordings_shuffled
                )
                loss_out_shuffled, reg_shuffled = self.get_metrics_classification(
                    m_target_shuffled, target, spk_rec_target_shuffled, "val_shuffled"
                )
        elif (
            self.model.training_mode == "trace_learning"
            or self.model.training_mode == "trace_learning_BLK"
        ):
            loss_out, reg = self.get_metrics_trace_learning(
                output,
                target_output,
                other_recordings,
                target_other_recordings,
                target,
                "val",
            )
            loss_out_shuffled, reg_shuffled = self.get_metrics_trace_learning(
                target_output_shuffled,
                target_output,
                target_other_recordings_shuffled,
                target_other_recordings,
                target,
                "val_shuffled",
            )

        loss_hidden = 0
        if self.target_model is not None:
            loss_hidden = self.get_additional_metrics(
                self.model,
                output,
                target_output,
                other_recordings,
                target_other_recordings,
                "val",
            )
            loss_hidden_shuffled = self.get_additional_metrics(
                self.target_shuffled,
                target_output_shuffled,
                target_output,
                target_other_recordings_shuffled,
                target_other_recordings,
                "val_shuffled",
            )

        loss = reg
        if self.train_hidden:
            loss += loss_hidden
        if self.train_out:
            loss += loss_out

        self.log("val_loss", loss)
        self.log_weights()  # only logged here because validation is less frequent than training
        if not self.model.is_student and not self.model.train_weights:
            self.log("weight_scaling_factor", self.model.weight_scaling_factor)

        # free storage
        torch.cuda.empty_cache()
        gc.collect()

    def plot_mem_hidden(self, mem_rec, spk_rec):
        fig = plt.figure(dpi=100)
        if self.model.is_spiking:
            plot_voltage_traces(mem_rec, spk=spk_rec, name=self.model.neuron_name)
        else:
            plot_voltage_traces(mem_rec, name=self.model.neuron_name)
        img = fig2img(plt)
        if not self.call_outside_loop:
            self.logger.log_image("mem_hidden", img)
        plt.show()
        plt.close()

    def plot_mem_out(self, out_rec):
        fig = plt.figure(dpi=100)
        plot_voltage_traces(out_rec, name=self.model.neuron_name)
        # print(self.model.last_h1.shape)
        img = fig2img(plt)
        if not self.call_outside_loop:
            self.logger.log_image("mem_out", img)
        plt.show()
        plt.close()

    def plot_v(self):
        if not self.model.v.is_sparse:
            if self.target_model is not None:
                v = self.target_model.v.detach().cpu().numpy()
                abs_max = np.abs(v).max()
                fig = plt.imshow(v, cmap="seismic", vmin=-abs_max, vmax=abs_max)
                plt.title("weight value")
                plt.colorbar(fig)
                img = fig2img(plt)
                if not self.call_outside_loop:
                    self.logger.log_image("weight_v_teacher", img)
                plt.show()
                plt.close()

                if self.target_shuffled is not None:
                    v = self.target_shuffled.v.detach().cpu().numpy()
                    abs_max = np.abs(v).max()
                    fig = plt.imshow(v, cmap="seismic", vmin=-abs_max, vmax=abs_max)
                    plt.title("weight value")
                    plt.colorbar(fig)
                    img = fig2img(plt)
                    if not self.call_outside_loop:
                        self.logger.log_image("weight_v_teacher_shuffled", img)
                    plt.show()
                    plt.close()

            v = self.model.v.detach().cpu().numpy()
            abs_max = np.abs(v).max()
            fig = plt.imshow(v, cmap="seismic", vmin=-abs_max, vmax=abs_max)
            plt.title("weight value")
            plt.colorbar(fig)
            img = fig2img(plt)
            if not self.call_outside_loop:
                self.logger.log_image("weight_v", img)
            plt.show()
            plt.close()

            try:
                fig = plt.imshow(
                    self.model.sparse_mask.numpy(),
                    cmap="seismic",
                    vmin=-abs_max,
                    vmax=abs_max,
                )
                plt.title("sparse mask")
                plt.colorbar(fig)
                img = fig2img(plt)
                if not self.call_outside_loop:
                    self.logger.log_image("sparse_mask", img)
                plt.show()
                plt.close()
            except:
                pass

            v_flat = v.flatten()
            fig = plt.hist(v_flat)
            plt.xlabel("weight value")
            plt.ylabel("count")
            plt.title("weight value distr")
            img = fig2img(plt)
            if not self.call_outside_loop:
                self.logger.log_image("weight_v_dist", img)
            plt.show()
            plt.close()

            # plot degree distributions for in and out
            degrees_in = np.sum(v, axis=0)
            degrees_out = np.sum(v, axis=1)
            fig = plt.hist(degrees_in)
            plt.xlabel("in-degree")
            plt.ylabel("neuron count")
            plt.title("in-degree distr")
            img = fig2img(plt)
            if not self.call_outside_loop:
                self.logger.log_image("in_deg", img)
            plt.show()
            plt.close()

            fig = plt.hist(degrees_out)
            plt.xlabel("out-degree")
            plt.ylabel("neuron count")
            plt.title("out-degree distr")
            img = fig2img(plt)
            if not self.call_outside_loop:
                self.logger.log_image("out_deg", img)
            plt.show()
            plt.close()

    def plot_reset(self):
        if "reset" in self.model.train_intrinsic:
            fig = plt.figure(dpi=100)

            width = 0.04

            x = torch.load(f"{self.model_path}/init_reset").detach().numpy()
            y = self.model.reset.cpu().detach().numpy()

            # bins = np.linspace(-1, 1.1, 15)

            N_hom_bins = int((np.sum(y[y > 0.5]) / 128) * 20)
            N_snic_bins = 20 - N_hom_bins
            snic_color = "#33cccc"  # Hesse light blue
            hom_color = "#cc0000"  # Hesse red
            mixed_color = "#0000ff"  # latex blue
            other_color = "#2da02c"  # green
            lif_color = "gray"

            plt.hist(
                x - width / 2,
                20,
                width=width,
                label="init",
                color="gray",
                align="mid",
                edgecolor="black",
                linewidth=1.5,
            )

            if "LIF" in self.model.neuron_name:
                plt.hist(
                    y - width / 2,
                    bins=20,
                    width=width,
                    label="post",
                    color=lif_color,
                    edgecolor=lif_color,
                    align="mid",
                )
            else:
                if N_hom_bins > 0:
                    plt.hist(
                        y[y > 0.5] - width / 2,
                        width=width,
                        bins=N_hom_bins,
                        color=hom_color,
                        align="mid",
                    )
                if N_snic_bins > 0:
                    plt.hist(
                        y[y <= 0.5] - width / 2,
                        width=width,
                        bins=N_snic_bins,
                        color=snic_color,
                        align="mid",
                    )
                plt.axvline(x=0.5, color=other_color, linestyle="--")

            plt.title(self.model.neuron_name + " init")
            plt.xlabel(r"$V_{reset}$")
            plt.ylabel("count")
            plt.xlim(-0.2, 1.2)

            plt.grid(axis="y", zorder=-1.0, color="lightgray", alpha=0.3)
            sns.despine()

            plt.legend(loc="upper right")

            # print(self.model.last_h1.shape)
            img = fig2img(plt)
            if not self.call_outside_loop:
                self.logger.log_image("reset_distr", img)
            plt.show()
            plt.close()

    def plot_activities_student_teacher(
        self, spk_rec, target_spk_rec, mem_rec, target_mem_rec, output, target_output
    ):

        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        sample_idx = 0

        # hidden spikes
        _, time, neurons = spk_rec.shape
        dt = self.model.time_step
        axs[0, 0].imshow(
            spk_rec.detach()
            .cpu()
            .numpy()[sample_idx, :, :][:, self.model.sort_rec_unrec]
            .T,
            cmap="binary",
            aspect="auto",
        )
        for pos in ["right", "top"]:
            plt.gca().spines[pos].set_visible(False)
        axs[0, 0].set_ylabel("neuron")
        axs[0, 0].set_yticks(
            np.linspace(0, neurons - 1, num=6),
            np.linspace(1, neurons, num=6).astype(int),
        )
        axs[0, 0].set_xticks(
            np.linspace(0, time, num=6), np.round(np.linspace(0, time * dt, num=6), 1)
        )
        axs[0, 0].set_xlabel("time [s]")
        axs[0, 0].set_title(f"student: hidden neuron spikes")

        axs[0, 1].imshow(
            target_spk_rec.detach()
            .cpu()
            .numpy()[sample_idx, :, :][:, self.model.sort_rec_unrec]
            .T,
            cmap="binary",
            aspect="auto",
        )
        for pos in ["right", "top"]:
            plt.gca().spines[pos].set_visible(False)
        axs[0, 1].set_ylabel("neuron")
        axs[0, 1].set_yticks(
            np.linspace(0, neurons - 1, num=6),
            np.linspace(1, neurons, num=6).astype(int),
        )
        axs[0, 1].set_xticks(
            np.linspace(0, time, num=6), np.round(np.linspace(0, time * dt, num=6), 1)
        )
        axs[0, 1].set_xlabel("time [s]")
        axs[0, 1].set_title(f"teacher: hidden neuron spikes")

        # hidden voltages
        vmin, vmax = target_spk_rec.min(), target_spk_rec.max()

        axs[1, 0].imshow(
            mem_rec.detach()
            .cpu()
            .numpy()[sample_idx, :, :][:, self.model.sort_rec_unrec]
            .T,
            cmap="viridis",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        for pos in ["right", "top"]:
            plt.gca().spines[pos].set_visible(False)
        axs[1, 0].set_ylabel("neuron")
        axs[1, 0].set_yticks(
            np.linspace(0, neurons - 1, num=6),
            np.linspace(1, neurons, num=6).astype(int),
        )
        axs[1, 0].set_xticks(
            np.linspace(0, time, num=6), np.round(np.linspace(0, time * dt, num=6), 1)
        )
        axs[1, 0].set_xlabel("time [s]")
        axs[1, 0].set_title(f"student: hidden neuron voltages")

        axs[1, 1].imshow(
            target_mem_rec.detach()
            .cpu()
            .numpy()[sample_idx, :, :][:, self.model.sort_rec_unrec]
            .T,
            cmap="viridis",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        for pos in ["right", "top"]:
            plt.gca().spines[pos].set_visible(False)
        axs[1, 1].set_ylabel("neuron")
        axs[1, 1].set_yticks(
            np.linspace(0, neurons - 1, num=6),
            np.linspace(1, neurons, num=6).astype(int),
        )
        axs[1, 1].set_xticks(
            np.linspace(0, time, num=6), np.round(np.linspace(0, time * dt, num=6), 1)
        )
        axs[1, 1].set_xlabel("time [s]")
        axs[1, 1].set_title(f"teacher: hidden neuron voltages")

        # output voltages
        vmin, vmax = target_output.min(), target_output.max()
        _, time, neurons = output.shape
        axs[2, 0].imshow(
            output.detach().cpu().numpy()[sample_idx, :, :].T,
            cmap="viridis",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        for pos in ["right", "top"]:
            plt.gca().spines[pos].set_visible(False)
        axs[2, 0].set_ylabel("neuron")
        axs[2, 0].set_yticks(
            np.linspace(0, neurons - 1, num=6),
            np.linspace(1, neurons, num=6).astype(int),
        )
        axs[2, 0].set_xticks(
            np.linspace(0, time, num=6), np.round(np.linspace(0, time * dt, num=6), 1)
        )
        axs[2, 0].set_xlabel("time [s]")
        axs[2, 0].set_title(f"student: output neuron voltages")

        axs[2, 1].imshow(
            target_output.detach().cpu().numpy()[sample_idx, :, :].T,
            cmap="viridis",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        for pos in ["right", "top"]:
            plt.gca().spines[pos].set_visible(False)
        axs[2, 1].set_ylabel("neuron")
        axs[2, 1].set_yticks(
            np.linspace(0, neurons - 1, num=6),
            np.linspace(1, neurons, num=6).astype(int),
        )
        axs[2, 1].set_xticks(
            np.linspace(0, time, num=6), np.round(np.linspace(0, time * dt, num=6), 1)
        )
        axs[2, 1].set_xlabel("time [s]")
        axs[2, 1].set_title(f"teacher: output neuron voltages")

        # shade known recordings
        if self.model.hidden_in_loss is not None:
            for i in np.arange(2):
                for j in np.arange(2):
                    for k in np.arange(len(self.model.hidden_in_loss)):
                        axs[i, j].axhspan(k - 0.5, k + 0.5, facecolor="gray", alpha=0.5)

        plt.tight_layout()

        img = fig2img(plt)
        if not self.call_outside_loop:
            self.logger.log_image("activity_student_teacher", img)
        plt.show()
        plt.close()

    def plot_intr_bet_I(self, mode="neutral"):
        # matplotlib can otherwise cause problem when running on local mac
        matplotlib.use("agg")

        fig = plt.figure()
        gs = GridSpec(4, 4)
        ax_scatter = fig.add_subplot(gs[1:4, 0:3])
        ax_hist_x = fig.add_subplot(gs[0, 0:3])
        ax_hist_y = fig.add_subplot(gs[1:4, 3])
        xs = []
        ys = []
        colors = ["green", "red"]
        i = 0
        if mode == "neutral":
            curr_model = self.model
        elif mode == "teacher":
            curr_model = self.target_model
        elif mode == "student":
            curr_model = self.model

        if curr_model.beta is not None:
            beta = curr_model.beta
            x = (
                (-self.model.time_step / torch.log(beta)).cpu().detach().numpy()
            )  # tau_mem
        else:
            x = curr_model.tau_mem.cpu().detach().numpy()
        y = curr_model.I_c.cpu().detach().numpy()

        bins = np.linspace(-1, 1.1, 15)
        if mode == "neutral":
            c = np.arange(self.model.N_unique_ct)
            ax_scatter.scatter(x, y, c=c, alpha=0.3, cmap=plt.cm.coolwarm)
        elif mode == "teacher" or (
            (not "beta" in self.model.train_intrinsic)
            and (not "I_c" in self.model.train_intrinsic)
        ):
            ax_scatter.scatter(x, y, color=colors[i], alpha=0.3)
        elif mode == "student":
            teacher_model = self.target_model
            if teacher_model.beta is not None:
                beta = teacher_model.beta
                xt = (
                    (-self.model.time_step / torch.log(beta)).cpu().detach().numpy()
                )  # tau_mem
            else:
                xt = teacher_model.tau_mem.cpu().detach().numpy()
            yt = teacher_model.I_c.cpu().detach().numpy()

            x_error = x - xt
            y_error = y - yt
            error = np.sqrt(x_error**2 + y_error**2)
            divnorm = clr.TwoSlopeNorm(vcenter=0.0, vmax=4.25)
            ax_scatter.scatter(
                x, y, c=error, alpha=0.3, cmap=plt.cm.coolwarm, norm=divnorm
            )

        ax_hist_x.hist(x, color=colors[i], alpha=0.5)
        ax_hist_y.hist(y, orientation="horizontal", color=colors[i], alpha=0.5)

        ax_scatter.set_ylim(-1.1, 1.1)
        ax_scatter.set_xlim(-0.1, 1.1)
        ax_hist_x.set_xlim(-0.1, 1.1)
        # ax_hist_x.set_ylim(-0.1,1.1)
        ax_hist_y.set_xlim(0.0, 20)
        ax_hist_y.set_ylim(-1.1, 1.1)
        ax_scatter.set_xlabel(r"$\tau_m$")
        ax_scatter.set_ylabel(r"$I_c$", rotation="horizontal")
        plt.suptitle(r"joint distribution of $\tau_m$ and $I_c$")
        plt.tight_layout()
        img = fig2img(plt)
        if not self.call_outside_loop:
            name = "distr_intrinsic"
            if mode != "neutral":
                name += f"_{mode}"
            self.logger.log_image(name, img)
        plt.show()
        plt.close()

    def plot_intrinsic_distr(self):
        if self.target_model is None:
            self.plot_intr_bet_I(mode="neutral")
        else:
            self.plot_intr_bet_I(mode="teacher")
            self.plot_intr_bet_I(mode="student")

    def print_stats(self, spk_rec, m, target):
        print(f"spikes per neuron: {spk_rec.sum(axis=1).sum(axis=0)}")
        print(f"total spikes: {spk_rec.sum(axis=1).sum(axis=0).sum()}")
        print(f"w1 mean {self.model.w1.mean()}, w2 mean {self.model.w2.mean()}")
        print(f"w1 std {self.model.w1.std()}, w2 std {self.model.w2.std()}")
        N = target.size()[0]
        acc = self.get_acc(m, target, N)
        loss = self.get_loss_out(m, target, N)
        print(f"loss: {loss}, acc:{acc}")

    def test_step(self, batch, batch_idx, plot=False, print_stats=False):
        """ """
        data, target, duration = batch

        output, other_recordings = self.forward(data, target)
        mem_rec, spk_rec = other_recordings

        m, _ = torch.max(output, 1)
        _, am = torch.max(m, 1)  # argmax over output units

        if (
            self.target_model is not None
        ):  # evaluate batch with target model for metrics, even if NLL classification used
            target_output, target_other_recordings = self.target_forward(data, target)
            m_target, _ = torch.max(target_output, 1)
            _, am_target = torch.max(m_target, 1)  # argmax over output units
            target_output_shuffled, target_other_recordings_shuffled = (
                self.target_forward_shuffled(data, target)
            )
            m_target_shuffled, _ = torch.max(target_output_shuffled, 1)

        if (
            self.model.training_mode == "classification"
            or self.model.training_mode == "classification_BLK"
        ):
            mem_rec, spk_rec = other_recordings
            m, _ = torch.max(output, 1)
            loss_out, reg = self.get_metrics_classification(m, target, spk_rec, "test")
            if self.target_model is not None:
                mem_rec_target_shuffled, spk_rec_target_shuffled = (
                    target_other_recordings_shuffled
                )
                loss_out_shuffled, reg_shuffled = self.get_metrics_classification(
                    m_target_shuffled, target, spk_rec_target_shuffled, "test_shuffled"
                )
        elif (
            self.model.training_mode == "trace_learning"
            or self.model.training_mode == "trace_learning_BLK"
        ):
            loss_out, reg = self.get_metrics_trace_learning(
                output,
                target_output,
                other_recordings,
                target_other_recordings,
                target,
                "test",
            )
            loss_out_shuffled, reg_shuffled = self.get_metrics_trace_learning(
                target_output_shuffled,
                target_output,
                target_other_recordings_shuffled,
                target_other_recordings,
                target,
                "test_shuffled",
            )

        loss_hidden = 0
        if self.target_model is not None:
            loss_hidden = self.get_additional_metrics(
                self.model,
                output,
                target_output,
                other_recordings,
                target_other_recordings,
                "test",
            )
            loss_hidden_shuffled = self.get_additional_metrics(
                self.target_shuffled,
                target_output_shuffled,
                target_output,
                target_other_recordings_shuffled,
                target_other_recordings,
                "test_shuffled",
            )

        loss = reg
        if self.train_hidden:
            loss += loss_hidden
        if self.train_out:
            loss += loss_out

        if not self.call_outside_loop:
            self.log("test_loss", loss)

        if not self.model.is_student:
            self_shuffled = copy.deepcopy(self.model).eval()
            shuffle_idx = np.arange(self.model.nb_hidden)
            np.random.shuffle(shuffle_idx)
            self_shuffled.w1.data = self_shuffled.w1.data[:, shuffle_idx]
            self_shuffled.w2.data = self_shuffled.w2.data[shuffle_idx, :]
            self_shuffled.v.data = self_shuffled.v.data[shuffle_idx, :]
            self_shuffled.v.data = self_shuffled.v.data[:, shuffle_idx]

            out_rec_shuffled, other_recs_shuffled = self_shuffled.forward(data, target)
            m_shuffled, _ = torch.max(out_rec_shuffled, 1)
            N = target.size()[0]
            acc_shuffled = self.get_acc(m_shuffled, target, N)
            if not self.call_outside_loop:
                self.log("test_shuffled_acc", acc_shuffled)

        self.plot_mem_hidden(mem_rec, spk_rec)
        self.plot_mem_out(output)
        self.plot_reset()
        self.plot_v()
        self.plot_intrinsic_distr()

        if plot:
            np.save(f"spikes_hidden", spk_rec.cpu().detach().numpy())
            np.save(f"mem_hidden", mem_rec.cpu().detach().numpy())
            np.save(f"mem_out", output.cpu().detach().numpy())
            np.save("label", target.cpu().detach().numpy())

        if print_stats:
            self.print_stats(spk_rec, m, target)

        if self.call_outside_loop:
            np.save("data", data.cpu().detach().numpy())
            np.save("target", target.cpu().detach().numpy())
            np.save("output", output.cpu().detach().numpy())
            np.save("mem_rec", mem_rec.cpu().detach().numpy())
            np.save("spk_rec", spk_rec.cpu().detach().numpy())
            np.save("target_output", target_output.cpu().detach().numpy())
            target_mem_rec, target_spk_rec = target_other_recordings
            np.save("target_mem_rec", target_mem_rec.cpu().detach().numpy())

        self.all_preds.append(am.cpu())
        self.all_targets.append(target.cpu())
        if self.target_model is not None:
            self.all_preds_teacher.append(am_target.cpu())

    def on_test_epoch_end(self):

        all_preds = torch.cat(self.all_preds)

        all_targets = torch.cat(self.all_targets)
        self.get_confusion_matrix(all_preds, all_targets, "conf_student")

        if self.target_model is not None:
            all_preds_teacher = torch.cat(self.all_preds_teacher)
            self.get_confusion_matrix(all_preds_teacher, all_targets, "conf_teacher")

        # free storage
        torch.cuda.empty_cache()
        gc.collect()

    def get_confusion_matrix(self, pred, target, name):

        conf = confusion_matrix(pred, target, normalize="pred")
        print(f"pred max is {pred.max()}, target max is {target.max()}")
        print(f"pred shape is {pred.shape}, target sha[e] is {target.shape}")
        print(f"conf is {conf}")
        N = len(conf)
        classes = np.arange(N)
        plt.imshow(conf, cmap="Blues")
        for (i, j), z in np.ndenumerate(conf):
            plt.text(j, i, np.round(z, 2), ha="center", va="center")
        plt.xlabel("predicted class")
        plt.ylabel("target class")
        plt.yticks(classes, classes + 1)
        plt.xticks(classes, classes + 1)  # ,rotation=90)
        img = fig2img(plt)
        if not self.call_outside_loop:
            self.logger.log_image(name, img)
        plt.show()
        plt.close()
