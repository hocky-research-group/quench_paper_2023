#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=quench_umbrella_psi

cd $PWD
module purge
source initial.sh

python run_quench_umbrella_alanine_parsl.py parameters.yaml
python infinite_stopping_compute_lnrho_list.py parameters.yaml
python infinite_stopping_compute_N_list.py parameters.yaml
python quench_wham2d_partial.py parameters.yaml
python quench_wham2d_partial_N.py parameters.yaml

python run_quench_umbrella_alanine_parsl.py parameters_highT.yaml
python infinite_stopping_compute_lnrho_list.py parameters_highT.yaml
python quench_wham2d_partial.py parameters_highT.yaml

