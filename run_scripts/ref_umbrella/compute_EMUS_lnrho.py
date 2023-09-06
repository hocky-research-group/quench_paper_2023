import numpy as np
import os
import glob
from quench_library import angle_distance2_trj_pbc,compute_lnrho_EMUS_2d,log_sum

# load parameters
import yaml,sys
if len(sys.argv) != 2 or sys.argv[1][-5:] != ".yaml":
    print("Usage %s yaml_file"%(sys.argv[0]))
    sys.exit(-1)
with open(sys.argv[1],'r') as f:
    parameters = yaml.full_load(f)
    high_T_params = parameters["high_T"]
    umbrella_params = parameters["umbrella"]
    analysis_params = parameters["analysis"]
    globals().update(high_T_params)
    globals().update(umbrella_params)
    globals().update(analysis_params)
dphi = 2.0 * np.pi / phi_windows
dpsi = 2.0 * np.pi / psi_windows
phi_centers = np.arange(-np.pi+dphi/2.0,np.pi,dphi)
psi_centers = np.arange(-np.pi+dpsi/2.0,np.pi,dpsi)
fes_dphi = 2.0 * np.pi / fes_phi_windows
fes_dpsi = 2.0 * np.pi / fes_psi_windows
fes_phi_centers = np.arange(-np.pi+fes_dphi/2.0,np.pi,fes_dphi)
fes_psi_centers = np.arange(-np.pi+fes_dpsi/2.0,np.pi,fes_dpsi)
run_temp_list = [300.0]


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
                max_blocks=40,
                partition='',
                scheduler_options='#SBATCH --mem=8GB',
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
run_analysis = python_app(compute_lnrho_EMUS_2d)

for run_temp in run_temp_list:
    run_kbt = run_temp * kb
    in_dir_prefix = os.path.join(os.getcwd(),"high_T_umbrella","T%.1f"%(run_temp))
    out_dir_prefix = os.path.join(os.getcwd(),"EMUS_analysis","T%.1f"%(run_temp))
    if not os.path.exists(out_dir_prefix):
        os.makedirs(out_dir_prefix,exist_ok=True)
    result_list = []
    lnrho_matrix = np.zeros((phi_windows,psi_windows,fes_phi_windows,fes_psi_windows))
    lnone_matrix = np.zeros((phi_windows,psi_windows))
    lnF_matrix = np.zeros((phi_windows,psi_windows,phi_windows*psi_windows))
    for i,phi in enumerate(phi_centers):
        for j,psi in enumerate(psi_centers):
            in_dir = os.path.join(in_dir_prefix,"phi%.2f_psi%.2f_k%.1f"%(phi,psi,kappa))
            out_dir = os.path.join(out_dir_prefix,"phi%.2f_psi%.2f_k%.1f"%(phi,psi,kappa))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir,exist_ok=True)
            restart_files = glob.glob(os.path.join(in_dir,"alanine_langevin_rt%.1f_rg%.2e_eqsteps%d_step%d.restart"%(run_temp,run_gamma,eq_steps,run_steps)))
            for restart_file in restart_files: # only one log file
                log_file = restart_file.replace("_step%d.restart"%(run_steps),".log")
                r = run_analysis(log_file,phi_windows,psi_windows,fes_phi_windows,fes_psi_windows,kappa,run_kbt,i,j)
                result_list.append(r)
    for r in result_list:
        try:
            result = r.result()
            i = result[0]
            j = result[1]
            out_dir = os.path.join(out_dir_prefix,"phi%.2f_psi%.2f_k%.1f"%(phi_centers[i],psi_centers[j],kappa))
            lnrho = result[2]
            lnone = result[3]
            lnF = result[4]
            np.save(os.path.join(out_dir,"lnrho.npy"),lnrho)
            np.save(os.path.join(out_dir,"lnone.npy"),lnone)
            np.save(os.path.join(out_dir,"lnF.npy"),lnF)
            lnrho_matrix[i,j] = lnrho
            lnone_matrix[i,j] = lnone
            lnF_matrix[i,j] = lnF
        except Exception as e:
            print(f"Application {r} failed.")
    out_dir = os.path.join(os.getcwd(),"EMUS","T%.1f"%(run_temp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    np.save(os.path.join(out_dir,"lnrho_matrix.npy"),lnrho_matrix.reshape(phi_windows*psi_windows,fes_phi_windows,fes_psi_windows))
    np.save(os.path.join(out_dir,"lnone_matrix.npy"),lnone_matrix.reshape(phi_windows*psi_windows))
    np.save(os.path.join(out_dir,"lnF_matrix.npy"),lnF_matrix.reshape(phi_windows*psi_windows,phi_windows*psi_windows))

