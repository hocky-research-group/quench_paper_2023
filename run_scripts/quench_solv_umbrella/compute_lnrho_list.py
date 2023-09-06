from quench_library import compute_lnrho_2d,log_sum
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
    analysis_params = parameters["analysis"]
    globals().update(high_T_params)
    globals().update(umbrella_params)
    globals().update(quench_params)
    globals().update(analysis_params)
dphi = 2.0 * np.pi / phi_windows
dpsi = 2.0 * np.pi / psi_windows
phi_centers = np.arange(-np.pi+dphi/2.0,np.pi,dphi)
psi_centers = np.arange(-np.pi+dpsi/2.0,np.pi,dpsi)
fes_dphi = 2.0 * np.pi / fes_phi_windows
fes_dpsi = 2.0 * np.pi / fes_psi_windows
fes_phi_centers = np.arange(-np.pi+fes_dphi/2.0,np.pi,fes_dphi)
fes_psi_centers = np.arange(-np.pi+fes_dpsi/2.0,np.pi,fes_dpsi)
dof = natoms * 3 

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
                init_blocks=100,
                min_blocks=100,
                max_blocks=100,
                partition='',
                scheduler_options='', #'#SBATCH --mem=8GB',
                worker_init='module purge; source %s'%(os.path.join(os.getcwd(),"initial.sh")),
                launcher=SrunLauncher(),
                exclusive=False,
                walltime='96:00:00'
            ),
       )
    ],
    retries=0,
)
parsl.load(config)
run_analysis = python_app(compute_lnrho_2d)

run_kbt = run_temp * kb
target_kbt = target_temp * kb
dgT = np.log(run_temp / 300.0)
gT = gT0 + dgT
gT_b = gT0_b - dgT

in_dir_prefix = os.path.join(os.getcwd(),"quench","T%.1f"%(run_temp))
out_dir_prefix = os.path.join(os.getcwd(),"analysis","T%.1f"%(run_temp))
if not os.path.exists(out_dir_prefix):
    os.makedirs(out_dir_prefix,exist_ok=True)
for quench_gamma in quench_gamma_list:
    for phi in phi_centers:
        for psi in psi_centers:
            result_list = []
            in_dir = os.path.join(in_dir_prefix,"phi%.2f_psi%.2f_k%.1f"%(phi,psi,kappa))
            out_dir = os.path.join(out_dir_prefix,"phi%.2f_psi%.2f_k%.1f"%(phi,psi,kappa))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir,exist_ok=True)
            restart_files = glob.glob(os.path.join(in_dir,"alanine_langevin_rt%.1f_rg%.2e_eqsteps%d_step*_gT%d_gTb%d_qg%.2e_step%d.restart"%(run_temp,run_gamma,eq_steps,gT,gT_b,quench_gamma,int(gT/quench_gamma/dt))))
            print("phi%.2f,psi%.2f,kappa%.1f,rt%.1f,tt%.1f,%d restart files"%(phi,psi,kappa,run_temp,target_temp,len(restart_files)))
            for i in range(len(restart_files)):
                log_file = os.path.join(in_dir,"alanine_langevin_rt%.1f_rg%.2e_eqsteps%d_step%d_gT%d_gTb%d_qg%.2e.log"%(run_temp,run_gamma,eq_steps,run_steps//nrestart*(i+1),gT,gT_b,quench_gamma))
                r = run_analysis(log_file,fes_phi_windows,fes_psi_windows,target_kbt,run_kbt,dof,dt,quench_gamma,i,heat=True)
                result_list.append(r)
            lnrho_list = np.zeros((len(restart_files),fes_phi_windows,fes_psi_windows))
            for r in result_list:
                try:
                    result = r.result()
                    lnrho_list[result[0]] = result[1]
                    print(f"Got results: {result}.")
                except Exception as e:
                    print(f"Application {r} failed but continuing.")
            # save data
            lnrho_list = np.array(lnrho_list)
            np.save(os.path.join(out_dir,"lnrho_list_phi%.2f_psi%.2f_k%.1f_qg%.2e_tt%.1f_%dx%d.npy"%(phi,psi,kappa,quench_gamma,target_temp,fes_phi_windows,fes_psi_windows)),lnrho_list)
            lnrho = np.zeros((fes_phi_windows,fes_psi_windows))
            for i in range(fes_phi_windows):
                for j in range(fes_psi_windows):
                    lnrho[i,j] = log_sum(lnrho_list[:,i,j])
            lnrho -= np.log(len(lnrho_list))
            np.save(os.path.join(out_dir,"lnrho_phi%.2f_psi%.2f_k%.1f_qg%.2e_tt%.1f_%dx%d.npy"%(phi,psi,kappa,quench_gamma,target_temp,fes_phi_windows,fes_psi_windows)),lnrho)

