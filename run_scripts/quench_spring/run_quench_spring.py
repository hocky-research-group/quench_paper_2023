import numpy as np
import os
import glob
from quench_library import run_quench_spring

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
            max_workers=1, # Treat this as max_worker_per_node. Since you have a single core application, this should be # of cores
            cores_per_worker=1,
            provider=SlurmProvider(
                # In Parsl block == Scheduler JOB
                nodes_per_block=1,
                init_blocks=100,
                min_blocks=1,
                max_blocks=100,
                partition='',
                scheduler_options='', #'#SBATCH --mem=8GB',
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
run_quench = python_app(run_quench_spring)

# yaml
import yaml,sys
if len(sys.argv) != 2 or sys.argv[1][-5:] != ".yaml":
    print("Usage: python %s yaml_file"%(sys.argv[0]))
with open(sys.argv[1]) as f:
    parameters = yaml.full_load(f)
    high_T_params = parameters["high_T"]
    quench_params = parameters["quench"]
    globals().update(high_T_params)
    globals().update(quench_params)

in_dir_prefix = os.path.join(os.getcwd(),"high_T")
out_dir_prefix = os.path.join(os.getcwd(),"quench")
if not os.path.exists(out_dir_prefix):
    os.makedirs(out_dir_prefix,exist_ok=True)
result_list = []
for N in N_list:
    for run_temp in run_temp_list:
        gT = gT0 + np.log(run_temp)
        gT_b = gT0_b - np.log(run_temp)
        in_dir = os.path.join(in_dir_prefix,"N%d"%(N),"T%.1f"%(run_temp))
        out_dir = os.path.join(out_dir_prefix,"N%d"%(N),"T%.1f"%(run_temp))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        restart_files = glob.glob(os.path.join(in_dir,"spring_N%d_K%.1f_rt%.1f_rg%.2e_eqsteps%d_step*.restart"%(N,kappa,run_temp,run_gamma,eq_steps)))
        print("rt%.1f,%d starting points"%(run_temp,len(restart_files)))
        for restart_file in restart_files:
            for quench_gamma in quench_gamma_list:
                log_prefix = os.path.join(out_dir,os.path.basename(restart_file).replace(".restart","_"))
                r = run_quench(command_file,restart_file,log_prefix,dt,kappa,quench_gamma,quench_thermo_freq,int(gT/quench_gamma/dt))
                result_list.append(r)
                r = run_quench(command_file,restart_file,log_prefix,dt,kappa,-quench_gamma,quench_thermo_freq,int(gT_b/quench_gamma/dt))
                result_list.append(r)
        
for r in result_list:
    try:
        result = r.result()
        print(f"Got result: {result}.")
    except Exception as e:
        print(f"Application {r} failed but continuing.")

