from quench_library import run_quench_alanine
import numpy as np
import os
import glob

# load parameters
import yaml,sys
if len(sys.argv) != 2 or sys.argv[1][-5:] != ".yaml":
    print("Usage %s yaml_file"%(sys.argv[0]))
    sys.exit(-1)
with open(sys.argv[1],'r') as f:
    parameters = yaml.full_load(f)
    high_T_params = parameters["high_T"]
    quench_params = parameters["quench"]
    globals().update(high_T_params)
    globals().update(quench_params)

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
                max_blocks=200,
                partition='',
                scheduler_options='', #'#SBATCH --mem=8GB',
                worker_init='module purge; source %s'%(os.path.join(os.getcwd(),"initial.sh")),
                launcher=SrunLauncher(),
                exclusive=False,
                walltime='48:00:00'
            ),
       )
    ],
    retries=0,
)
parsl.load(config)
run_quench = python_app(run_quench_alanine)

dgT = np.log(run_temp / 300.0)
gT = gT0 + dgT 
gT_b = gT0_b - dgT
in_dir = os.path.join(os.getcwd(),"high_T","T%.1f"%(run_temp))
out_dir = os.path.join(os.getcwd(),"quench","T%.1f"%(run_temp))
if not os.path.exists(out_dir):
    os.makedirs(out_dir,exist_ok=True)

result_list = []
restart_files = glob.glob(os.path.join(in_dir,"alanine_langevin_rt%.1f_rg%.2e_eqsteps%d_step*.restart"%(run_temp,run_gamma,eq_steps)))
print("%d restart files"%(len(restart_files)))

for restart_file in restart_files:
    for quench_gamma in quench_gamma_list:
        r = run_quench(command_file,restart_file,out_dir,quench_thermo_freq,quench_gamma,gT,gT_b,dt=dt,heat=False)
        result_list.append(r)
        r = run_quench(command_file,restart_file,out_dir,quench_thermo_freq*5,-quench_gamma,gT,gT_b,dt=0.2*dt,heat=True)
        result_list.append(r)

for r in result_list:
    try:
        result = r.result()
        print(f"Got results: {result}.")
    except Exception as e:
        print(f"Application {r} failed but continuing.")

