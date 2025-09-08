# Optimizing Connectome-constrained Spiking Neural Networks.

Repository for training weights and/or intrinsic parameters in SNNs and RNNs in a teacher/student setup for a hypothetical connectome-constrained scenario. Models can be optimized on classification tasks and on teacher's hidden or output layer voltages.

In the Nengo summer school, we have extended the model to the fruit fly olfactory pathway (instead of the 'teacher'). We used the codex connectome and neurotransmitter information to recreate the architecture and weight matrices, and created a new dataset for odor classification based on the DoOR dataset (Münch & Galizia, 2016). One of the first ever trained model already reaches a validation set accuracy of 76% after only 6 training epochs (chance level <0.36% for 284 classes) and we see much sparser neural activity in e.g. KC than in PN neurons.

The training, validation, and testing metrics are logged automatically using [Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) modules and Weights & Biases, and can be tracked online. Please make sure to log in to your Weights & Biases account and create a project 'CC_SNN' to log to ([W&B Quickstart](https://docs.wandb.ai/quickstart/)). We log losses and evaluation metric on training and validatipon set after every epoch, and on the test set after training. Model checkpoints are saved every 10 epochs.


## Next steps 

Most relevant related work: [Lappalainen et al. (2024)](https://www.nature.com/articles/s41586-024-07939-3) (connectome-constrained optimization without spikes), [Nanami et al. (2024)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1384336/full) (olfactory pathway on neuromorphic hardware), [Schlegel et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/34032214/) (Drosophila olfactory system)

1. Finalize model transfer to SpiNNaker for inference
2. Train multiple models (e.g. 20) on same task, furthermore
- log #spikes per cell type for better quantitative evaluation over training
- allow for training synaptic time constant per neurotransmitter
- investigate single-neuron traces in vivo vs in silico
- other versions
    - shuffled baseline (how to shuffle? -> shuffle weights within larger cell types)
    - allow training of weights (not just per nt, but for each synapse)
        - with connectome prior
        - without connectome prior

## Investigation ideas
Ideally, at the end of the summer school, we will have an SNN recovering function from the fruit fly odor pathway. The SNN's main task is odor classification, but this is used mainly to optimize parameters. After optimization, we can investigate in detail how similar the SNN and the biological counterpart are in terms of low-level activities, but also of higher-order properties. Here, we are collecting our ideas on which comparisons could be performed.

- check spiking pattern & time-averaged firing rate in silico vs in vivo (=Nanami et al. (2024) before associative learning)
- check tuning prolfiles of higher-order cells (not OSN, those are input) (Turner et al. 2017)
    - PN broadly tuned
    - in mushroom body (KC?): sparse encoding, small overlap between representations
- check if whitening is performed (Wanner & Friedrich, 2020, for zebrafish - probably also some Drosophila papers)
- most responsive KC types (Turner et al., 2008)
- predict innate attractive/aversive odors (hinted at by Campbell et al., 2013)
- associative learning; new task: free all weights, see if the weights of KC>MBON-α1 synapses are weakened? (Nanami et al.)
- parallel odors - some type of asynchrony? (Terry)
- role of APL neurons (Kathryn)

## Setup

If you plan to use pytorch cuda, please add `- pytorch-cuda=<appropriate-verison>` to `environment.yml`. Then, install environment, e.g.

```
conda env create -f environment.yml
                         
```
If possible, download [Pyspike](https://anaconda.org/conda-forge/pyspike) or copy the folder `Pyspike/pyspike/` [(Github link)](https://github.com/mariomulansky/PySpike/tree/master) into your repository.

For using the SpiNNaker hardware, please use

```
pip install –upgrade sPyNNaker
pip install PyNN
```
You will need to set your host IP address to 192.168.240.2 . Then you will need to make and 
Please also run RunForConfig.py . This will create a file called .spynnaker.cfg in your home folder (for me it was C:\Users\madta\.spynnaker.cfg). 
Then edit that file with the following parameters. machine name is because the spynnaker board is preset to this static IP address.
Version 3 corresponds to the smaller spinnaker board. See https://spinnakermanchester.github.io/spynnaker/8.0.0/PyNNOnSpinnakerInstall.html
machineName = 192.168.240.9
version = 3


## Data

### Connectome Data
To use Drosophila connectome information from [codex/FlyWire](https://codex.flywire.ai/api/download?dataset=fafb), first download the following files:
- `connections.csv.gz` (Connections (Filtered))
- `consolidated_cell_types.csv.gz` (Cell Types)
- `names.csv.gz` (Proofread Cell Names and Groups)
- `processed_labels.csv.gz` (Community Labels (Refined))
into `projects/data/connectome` and unzip each file by running
```
gunzip <filename>
```
from `projects/data/connectome`.
The `.txt` files in `projects/data/connectome/codex_keywords` are cell IDs downloaded from [codex/FlyWire](https://codex.flywire.ai/?dataset=fafb) by searching for the `<keyword>` for each file `root_ids_<keyword>.txt`. The data is preprocessed in `projects/data/connectome/prep_pathway.ipynb`.

### Task Data
To create an odor classification task, we adapted code from [Nanami et al. (2024)](https://github.com/tnanami/fly-olfactory-network-fpga/blob/main/00_preparation/generate_ORNcsv.ipynb) in `projects/data/ORN_data/generate_ORN_data.ipynb` and downloaded the normalized response matrix (mean odor receptor calcium responses to various odors) from [Münch & Galizia (2016)](https://zenodo.org/records/46554) into `projects/data/ORN_data/data_r/response.matrix.RData`. We filtered the odors to have at least 3 neurons responding to it and created data samples of 500ms duration with 1ms resolution. The odor is presented for random durations between 70-500ms starting at 0ms. The resulting dataset consists of spike responses of 2264 olfactory receptor neurons to 284 odors.

## Running Experiments

Since our experiments consist of many runs, we make use of W&B sweeps. To train networks with connectome information from the fruit fly olfactory pathway, run
```
python sweep_sweep_fruitfly.py
                         
```

Results will be logged to your W&B project and checkpoints and additional logs are saved in subfolders of `results/olfactory/`, depending on specified parameters.

The sweep ID is saved in `<teacher-or-student>_{data_set_name}.txt` file and can be used to run models in parallel for the same sweep. Start a sweep without arguments or with `-0`, and log to the same sweep ID by running the same python file with an argument >0, e.g.
```
python sweep_fruitfly.py -1
                         
```

To modify experiments, add new network settings in `algorithm/config/settings`. Then, overwrite or vary configurations in your sweep setup as in e.g. `sweep_sweep_fruitfly.py`.


## Directory Structure

`algorithm/` contains SNN class and Lightning class, neurons, synapses, data laoding, and config files.

`results/` contains model checkpoints and W&B logfiles

`evaluate/` contains e.g. visualization code.

`data/` contains data sets, both for connectome (`connectome/`) and task (`ORN_data/`).

`helpers/` contains code for parallel runs in a sweep and for finding teacher networks.

`model_example/` contains checkpoints of student models.

`pynn_models/` contains pynn models to be run on spiNNaker hardware

