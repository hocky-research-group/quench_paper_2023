from quench_library import run_quench_umbrella_alanine
from quench_library import make_plumed_file
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
    umbrella_params = parameters["umbrella"]
    quench_params = parameters["quench"]
    globals().update(high_T_params)
    globals().update(umbrella_params)
    globals().update(quench_params)
dpsi = 2.0 * np.pi / psi_windows
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
                max_blocks=200,
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
run_quench = python_app(run_quench_umbrella_alanine)

dgT = np.log(run_temp / 300.0)
gT = gT0 + dgT
gT_b = gT0_b - dgT
in_dir_prefix = os.path.join(os.path.dirname(os.getcwd()),"ref_umbrella_psi","high_T_umbrella","T%.1f"%(run_temp))
out_dir_prefix = os.path.join(os.getcwd(),"quench","T%.1f"%(run_temp))
if not os.path.exists(out_dir_prefix):
    os.makedirs(out_dir_prefix,exist_ok=True)
result_list = []
for psi in psi_centers:
    in_dir = os.path.join(in_dir_prefix,"psi%.2f_k%.1f"%(psi,kappa))
    restart_files = glob.glob(os.path.join(in_dir,"alanine_langevin_rt%.1f_rg%.2e_eqsteps%d_step*.restart"%(run_temp,run_gamma,eq_steps)))
    print("psi%.2f,kappa%.1f,%d restart files"%(psi,kappa,len(restart_files)))
    # make plumed.dat
    out_dir = os.path.join(out_dir_prefix,"psi%.2f_k%.1f"%(psi,kappa))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    plumed_file = os.path.join(out_dir,os.path.basename(plumed_template))
    if not os.path.exists(plumed_file):
        plumed_file = make_plumed_file(plumed_template,out_dir,psi,kappa)
    for quench_gamma in quench_gamma_list:
        for restart_file in restart_files:
            # continue
            # cool down
            final_restart_file = os.path.join(out_dir,os.path.basename(restart_file).replace(".restart","_gT%d_gTb%d_qg%.2e_step%d.restart"%(gT,gT_b,quench_gamma,int(gT/quench_gamma/dt))))
            if not os.path.exists(final_restart_file):
                r = run_quench(command_file,restart_file,out_dir,quench_thermo_freq,quench_gamma,gT,gT_b,dt=dt,heat=False,plumed_file=plumed_file)
                result_list.append(r)
            # heat up
            final_restart_file_b = os.path.join(out_dir,os.path.basename(restart_file).replace(".restart","_gT%d_gTb%d_qg%.2e_step%d.restart"%(gT,gT_b,-quench_gamma,int(gT_b/quench_gamma/dt))))
            if not os.path.exists(final_restart_file_b):
                r = run_quench(command_file,restart_file,out_dir,5*quench_thermo_freq,-quench_gamma,gT,gT_b,dt=0.2*dt,heat=True,plumed_file=plumed_file) # change restart_b if changing dt value!!
                result_list.append(r)

for r in result_list:
    try:
        result = r.result()
        #result = r
        print(f"Got results: {result}.")
    except Exception as e:
        print(f"Application {r} failed but continuing.")


