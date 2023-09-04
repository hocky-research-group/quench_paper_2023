from quench_library import find_Ebound,infinite_stopping_compute_lnrho_2d,log_sum
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
    analysis_params = parameters["analysis"]
    globals().update(high_T_params)
    globals().update(quench_params)
    globals().update(analysis_params)
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
run_analysis = python_app(infinite_stopping_compute_lnrho_2d)
run_findE = python_app(find_Ebound)

run_kbt = run_temp * kb
target_kbt = target_temp * kb
dgT = np.log(run_temp / 300.0)
gT = gT0 + dgT
gT_b = gT0_b - dgT
in_dir = os.path.join(os.getcwd(),"quench","T%.1f"%(run_temp))
out_dir = os.path.join(os.getcwd(),"analysis","T%.1f"%(run_temp))
if not os.path.exists(out_dir):
    os.makedirs(out_dir,exist_ok=True)
for quench_gamma in quench_gamma_list:
    restart_files = glob.glob(os.path.join(in_dir,"alanine_langevin_rt%.1f_rg%.2e_eqsteps%d_step*_gT%d_gTb%d_qg%.2e_step%d.restart"%(run_temp,run_gamma,eq_steps,gT,gT_b,-quench_gamma,int((gT_b)/quench_gamma/dt)*2)))
    print("rt%.1f,tt%.1f,%d restart files, shoud be %d files"%(run_temp,target_temp,len(restart_files),nrestart))

    Emin_list_file = os.path.join(out_dir,"Emin_list_qg%.2e.npy"%(quench_gamma))
    Emax_list_file = os.path.join(out_dir,"Emax_list_qg%.2e.npy"%(quench_gamma))
    if not os.path.exists(Emin_list_file) or not os.path.exists(Emax_list_file):
        result_list = []
        for i in range(nrestart):
            log_file = os.path.join(in_dir,"alanine_langevin_rt%.1f_rg%.2e_eqsteps%d_step%d_gT%d_gTb%d_qg%.2e.log"%(run_temp,run_gamma,eq_steps,run_steps//nrestart*(i+1),gT,gT_b,quench_gamma))
            r = run_findE(log_file,i,heat=True)
            result_list.append(r)
        Emin_list = np.ones(nrestart) * -np.inf
        Emax_list = np.ones(nrestart) * np.inf
        for r in result_list:
            try:
                result = r.result()
                Emin_list[result[0]] = result[1]
                Emax_list[result[0]] = result[2]
            except Exception as e:
                print(f"Application {r} failed but continuing.")
        np.save(Emin_list_file,Emin_list)
        np.save(Emax_list_file,Emax_list)

    Emin_list = np.load(Emin_list_file)
    Emax_list = np.load(Emax_list_file)
    Emin = Emin_list[Emin_list!=-np.inf].max()
    Emax = Emax_list[Emax_list!=np.inf].min()
    print(Emin,Emax)

    result_list = []
    for i in range(nrestart):
        log_file = os.path.join(in_dir,"alanine_langevin_rt%.1f_rg%.2e_eqsteps%d_step%d_gT%d_gTb%d_qg%.2e.log"%(run_temp,run_gamma,eq_steps,run_steps//nrestart*(i+1),gT,gT_b,quench_gamma))
        r = run_analysis(log_file,fes_phi_windows,fes_psi_windows,target_kbt,run_kbt,dof,dt,quench_gamma,Emin,Emax,i,heat=True)
        result_list.append(r)
    lnrho_list = np.ones((nrestart,fes_phi_windows,fes_psi_windows)) * -np.inf
    lnQ_list = np.ones(nrestart) * -np.inf
    time_list = np.zeros(nrestart)
    for r in result_list:
        try:
            result = r.result()
            lnrho_list[result[0]] = result[1]
            lnQ_list[result[0]] = result[2]
            time_list[result[0]] = result[3]
        except Exception as e:
            print(f"Application {r} failed but continuing.")

    # save data
    np.save(os.path.join(out_dir,"infinite_stopping_lnrho_list_qg%.2e_tt%.1f_%dx%d.npy"%(quench_gamma,target_temp,fes_phi_windows,fes_psi_windows)),lnrho_list)
    lnrho = np.ones((fes_phi_windows,fes_psi_windows)) * -np.inf
    for i in range(fes_phi_windows):
        for j in range(fes_psi_windows):
            lnrho[i,j] = log_sum(lnrho_list[:,i,j])
    np.save(os.path.join(out_dir,"infinite_stopping_lnrho_qg%.2e_tt%.1f_%dx%d.npy"%(quench_gamma,target_temp,fes_phi_windows,fes_psi_windows)),lnrho)
    np.save(os.path.join(out_dir,"infinite_stopping_lnQ_list_qg%.2e_tt%.1f_%dx%d.npy"%(quench_gamma,target_temp,fes_phi_windows,fes_psi_windows)),lnQ_list)
    np.save(os.path.join(out_dir,"infinite_stopping_lnQ_qg%.2e_tt%.1f_%dx%d.npy"%(quench_gamma,target_temp,fes_phi_windows,fes_psi_windows)),log_sum(lnQ_list))
    np.save(os.path.join(out_dir,"infinite_stopping_time_list_qg%.2e_tt%.1f_%dx%d.npy"%(quench_gamma,target_temp,fes_phi_windows,fes_psi_windows)),time_list)


