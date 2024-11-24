#! /bin/sh

#SBATCH --job-name=get_sae_repr_and_loss
#SBATCH --output=run.out # redirect stdout
#SBATCH --error=run.err # redirect stderr
#SBATCH --partition=studentkillable# killable  # gpu-a100-killable #
#SBATCH --time=1440 # max time (minutes) (1 day)
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --gpus=1 # GPUs in total
#SBATCH --nodelist=s-004,s-005

python3 source/get_sae_repr_and_loss.py --run_config configs/run_config_gemma.yaml
# python3 source/get_sae_repr_and_loss.py --run_config configs/run_config.yaml





# To run use: sbatch run.slurm
# To check if it is running and on which server use: squeue -u <your_username>
# To get outputs use: cat run.out
# To get errors (most outputs and tqdm) use: cat run.err
# run scancel <job_id> to cancel the job