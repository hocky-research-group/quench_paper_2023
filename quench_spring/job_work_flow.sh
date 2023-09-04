#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=quench_spring

cd $PWD
module purge
source initial.sh

python gen_initial_data.py parameters_T.yaml
python run_langevin_spring.py parameters_T.yaml
python run_quench_spring.py parameters_T.yaml
python infinite_stopping_compute_lnQ_list.py parameters_T.yaml
python compute_error_T.py parameters_T.yaml

python gen_initial_data.py parameters_N.yaml
python run_langevin_spring.py parameters_N.yaml
python run_quench_spring.py parameters_N.yaml
python infinite_stopping_compute_lnQ_list.py parameters_N.yaml
python compute_error_N.py parameters_N.yaml

