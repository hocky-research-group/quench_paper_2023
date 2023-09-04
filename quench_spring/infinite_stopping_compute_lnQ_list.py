import numpy as np
import os
import glob
from quench_library import infinite_stopping_compute_lnQ,log_sum,find_Ebound

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
run_analysis = python_app(infinite_stopping_compute_lnQ)
run_findE = python_app(find_Ebound)

# yaml
import yaml,sys
if len(sys.argv) != 2 or sys.argv[1][-5:] != ".yaml":
    print("Usage: python %s yaml_file"%(sys.argv[0]))
with open(sys.argv[1]) as f:
    parameters = yaml.full_load(f)
    high_T_params = parameters["high_T"]
    quench_params = parameters["quench"]
    analysis_params = parameters["analysis"]
    globals().update(high_T_params)
    globals().update(quench_params)
    globals().update(analysis_params)
target_kbt = target_temp * kb

in_dir_prefix = os.path.join(os.getcwd(),"quench")
out_dir_prefix = os.path.join(os.getcwd(),"analysis")
if not os.path.exists(out_dir_prefix):
    os.makedirs(out_dir_prefix,exist_ok=True)
for N in N_list:
    dof = 3.0 * N
    for run_temp in run_temp_list:
        run_kbt = run_temp * kb
        dgT = np.log(run_temp)
        gT = gT0 + dgT
        gT_b = gT0_b - dgT
        in_dir = os.path.join(in_dir_prefix,"N%d"%(N),"T%.1f"%(run_temp))
        out_dir = os.path.join(out_dir_prefix,"N%d"%(N),"T%.1f"%(run_temp))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for quench_gamma in quench_gamma_list:
            for i in range(1,1+nrestarts):
                restart_file = os.path.join(in_dir,"spring_N%d_K%.1f_rt%.1f_rg%.2e_eqsteps%d_step%d_qg%.2e_step%d.restart"%(N,kappa,run_temp,run_gamma,eq_steps,i*thermo_freq,quench_gamma,int(gT/quench_gamma/dt)))
                if not os.path.exists(restart_file):
                    print("File %s does not exist!"%restart_file)
                restart_file = os.path.join(in_dir,"spring_N%d_K%.1f_rt%.1f_rg%.2e_eqsteps%d_step%d_qg%.2e_step%d.restart"%(N,kappa,run_temp,run_gamma,eq_steps,i*thermo_freq,-quench_gamma,int(gT_b/quench_gamma/dt)))
                if not os.path.exists(restart_file):
                    print("File %s does not exist!"%restart_file)

            Emin_list_file = os.path.join(out_dir,"Emin_list_qg%.2e.npy"%(quench_gamma))
            Emax_list_file = os.path.join(out_dir,"Emax_list_qg%.2e.npy"%(quench_gamma))
            if not os.path.exists(Emin_list_file) or not os.path.exists(Emax_list_file):
                result_list = []
                for i in range(1,1+nrestarts):
                    log_file = os.path.join(in_dir,"spring_N%d_K%.1f_rt%.1f_rg%.2e_eqsteps%d_step%d_qg%.2e.log"%(N,kappa,run_temp,run_gamma,eq_steps,i*thermo_freq,quench_gamma))
                    r = run_findE(log_file,i,heat=True)
                    result_list.append(r)
                Emin_list = np.ones(nrestarts) * -np.inf
                Emax_list = np.ones(nrestarts) * np.inf
                for r in result_list:
                    try:
                        result = r.result()
                        Emin_list[result[0]] = result[1]
                        Emax_list[result[0]] = result[2]
                        #print(f"Got result: {result}.")
                    except Exception as e:
                        print(f"Application {r} failed but continuing.")
                np.save(Emin_list_file,Emin_list)
                np.save(Emax_list_file,Emax_list)
            Emin_list = np.load(Emin_list_file)
            Emax_list = np.load(Emax_list_file)
            Emin = Emin_list.max()
            Emax = Emax_list.min()
            print(Emin,Emax)

            result_list = []
            for i in range(1,1+nrestarts):
                log_file = os.path.join(in_dir,"spring_N%d_K%.1f_rt%.1f_rg%.2e_eqsteps%d_step%d_qg%.2e.log"%(N,kappa,run_temp,run_gamma,eq_steps,i*thermo_freq,quench_gamma))
                r = run_analysis(log_file,target_kbt,run_kbt,dof,dt,quench_gamma,Emin,Emax,i-1,heat=True)
                result_list.append(r)
            lnQ_list = np.ones(nrestarts) * -np.inf
            time_list = np.zeros(nrestarts)
            for r in result_list:
                try:
                    result = r.result()
                    lnQ_list[result[0]] = result[1]
                    time_list[result[0]] = result[2]
                    print(f"Got results: {result}.")
                except Exception as e:
                    print(f"Application {r} failed but continuing.")
            np.save(os.path.join(out_dir,"infinite_stopping_lnQ_list_N%d_K%.1f_rt%.1f_rg%.2e_qg%.2e.npy"%(N,kappa,run_temp,run_gamma,quench_gamma)),lnQ_list)
            np.save(os.path.join(out_dir,"infinite_stopping_lnQ_N%d_K%.1f_rt%.1f_rg%.2e_qg%.2e.npy"%(N,kappa,run_temp,run_gamma,quench_gamma)),log_sum(lnQ_list))
            np.save(os.path.join(out_dir,"infinite_stopping_time_list_N%d_K%.1f_rt%.1f_rg%.2e_qg%.2e.npy"%(N,kappa,run_temp,run_gamma,quench_gamma)),time_list)


