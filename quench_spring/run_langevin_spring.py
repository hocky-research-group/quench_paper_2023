import numpy as np
import os
from quench_library import run_baoab_spring

# parsl
import parsl
from parsl.app.app import python_app
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
slurm_workers = os.getenv("SLURM_JOB_CPUS_PER_NODE")
if slurm_workers is not None:
    max_workers = int(slurm_workers)
else:
    max_workers = 1 
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname
config = Config(
    executors=[
        HighThroughputExecutor(
            label='Prince_HTEX',
            address=address_by_hostname(),
            max_workers=1, # Treat this as max_worker_per_node.
            provider=SlurmProvider(
                # In Parsl block == Scheduler JOB
                nodes_per_block=1,
                init_blocks=1,
                min_blocks=1,
                max_blocks=100,
                partition='',
                scheduler_options='#SBATCH --mem=2GB',
                worker_init='module purge; source %s'%(os.path.join(os.getcwd(),"initial.sh")),
                launcher=SrunLauncher(),
                exclusive=False,
                walltime='24:00:00'
            ),
       )
    ],
    retries=0,
)
parsl.load(config)
run_baoab = python_app(run_baoab_spring)

# yaml
import yaml,sys
if len(sys.argv) != 2 or sys.argv[1][-5:] != ".yaml":
    print("Usage: python %s yaml_file"%(sys.argv[0]))
with open(sys.argv[1]) as f:
    parameters = yaml.full_load(f)
    high_T_params = parameters["high_T"]
    globals().update(high_T_params)

out_dir_prefix = os.path.join(os.getcwd(),"high_T")
if not os.path.exists(out_dir_prefix):
    os.makedirs(out_dir_prefix,exist_ok=True)
#print(out_dir)
result_list = []
for N in N_list:
    input_file = input_file_template.replace("_N_","%d"%(N))
    for run_temp in run_temp_list:
        out_dir = os.path.join(out_dir_prefix,"N%d"%(N),"T%.1f"%(run_temp))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_prefix = os.path.join(out_dir,"spring_")
        r = run_baoab(command_file,input_file,N,kappa,dt,thermo_freq,run_temp,run_gamma,eq_steps,out_prefix,nrestarts)
        result_list.append(r)
        
for r in result_list:
    try:
        result = r.result()
        print(f"Got result: {result}.")
    except Exception as e:
        print(f"Application {r} failed but continuing.")

