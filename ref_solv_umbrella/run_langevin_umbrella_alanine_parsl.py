from quench_library import run_langevin_umbrella_alanine
from quench_library import make_plumed_file
import numpy as np
import os

# load parameters
import yaml,sys
if len(sys.argv) != 2 or sys.argv[1][-5:] != ".yaml":
    print("Usage %s yaml_file"%(sys.argv[0]))
    sys.exit(-1)
with open(sys.argv[1],'r') as f:
    parameters = yaml.full_load(f)
    high_T_params = parameters["high_T"]
    umbrella_params = parameters["umbrella"]
    globals().update(high_T_params)
    globals().update(umbrella_params)
dphi = 2.0 * np.pi / phi_windows
dpsi = 2.0 * np.pi / psi_windows
phi_centers = np.arange(-np.pi+dphi/2.0,np.pi,dphi)
psi_centers = np.arange(-np.pi+dpsi/2.0,np.pi,dpsi)

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
                scheduler_options='#SBATCH --mem=4GB',
                worker_init='module purge; source %s'%(os.path.join(os.getcwd(),"initial.sh")),
                launcher=SrunLauncher(),
                exclusive=False,
                walltime='144:00:00'
            ),
       )
    ],
    retries=0,
)
parsl.load(config)
run_langevin = python_app(run_langevin_umbrella_alanine)

result_list = []
for run_temp in run_temp_list:
    out_dir = os.path.join(os.getcwd(),"high_T_umbrella","T%.1f"%(run_temp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    for phi in phi_centers:
        for psi in psi_centers:
            out_umbrella_dir = os.path.join(out_dir,"phi%.2f_psi%.2f_k%.1f"%(phi,psi,kappa))
            out_prefix = os.path.join(out_umbrella_dir,"alanine_")
            if not os.path.exists(out_umbrella_dir):
                os.makedirs(out_umbrella_dir,exist_ok=True)
            # make plumed.dat
            plumed_file = os.path.join(out_umbrella_dir,os.path.basename(plumed_template))
            if not os.path.exists(plumed_file):
                plumed_file = make_plumed_file(plumed_template,out_umbrella_dir,phi,psi,kappa,kappa)
            pull_plumed_file = None
            r = run_langevin(command_file,input_file,out_prefix,run_gamma,run_temp,eq_steps,thermo_freq,run_steps,dt=dt,nrestart=nrestart,plumed_file=plumed_file,pull_plumed_file=pull_plumed_file)
            result_list.append(r)

for r in result_list:
    try:
        result = r.result()
        #result = r
        print(f"Got results: {result}.")
    except Exception as e:
        print(f"Application {r} failed but continuing.")

