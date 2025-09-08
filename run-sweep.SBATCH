#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:rtx8000
#SBATCH --mem=50GB
#SBATCH --job-name=olf_SNN
#SBATCH --account=pr_125_tandon_advanced
#SBATCH --output=/home/mie8014/projects/SNNs/summerschool2025/outputs/output_sbatch_only.out # Log file for each job

#GREENE GREENE_GPU_MPS=YES

export SSL_CERT_FILE=/home/mie8014/projects/SNNs/cc_th_SNN/cacert.pem

module purge

hostname

for((i=0; i<1; i++)); do
	log_file="/home/mie8014/projects/SNNs/summerschool2025/outputs/output_${SLURM_JOB_ID}_${i}.out"
    singularity exec --nv \
	    --overlay /scratch/mie8014/my_envs/SNNl/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; \
    	wandb artifact cache cleanup 100MB & python sweep_fruitfly.py \$0" $((i + 0)) > "$log_file" 2>&1 &
done

wait


