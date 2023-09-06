#!/bin/bash
module purge
module load intel/19.1.2
module load cmake/3.18.4
module load openmpi/intel/4.0.5
module load python/intel/3.8.6
#module load gcc/10.2.0
export PATH=/scratch/work/hockygroup/software/lammps-github-quench/src:$PATH

export PLUMED_PATH=/scratch/work/hockygroup/software/plumed2-icc-Sept2020
#export PLUMED_PATH=/scratch/work/hockygroup/software/plumed2-gcc-serial
export PLUMED_KERNEL=$PLUMED_PATH/libplumedKernel.so
export PATH=$PLUMED_PATH/bin:/scratch/work/hockygroup/software/lammps/:$PATH
export C_INCLUDE_PATH=$PLUMED_PATH/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$PLUMED_PATH/lib:$LD_LIBRARY_PATH

#do in custom environment
#pyenv=~/pyenv/md-py3.8
#source $pyenv/bin/activate
#cd /scratch/work/hockygroup/software/lammps-github-quench/python && python install.py $pyenv/lib/python3.8/site-packages && cd -   
export PYTHONPATH=$PYTHONPATH:$PWD

export PYTHONUNBUFFERED=1
cd /scratch/work/hockygroup/software/lammps-github-quench/python && python install.py && cd -

