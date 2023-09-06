#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=ref_solv_umbrella

cd $PWD
module purge
source initial.sh

python run_langevin_umbrella_alanine_parsl.py parameters.yaml
python compute_rho_umbrella.py parameters.yaml
python umbrella_wham2d_partial.py parameters.yaml

