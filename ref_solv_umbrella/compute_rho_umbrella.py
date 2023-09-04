import numpy as np
import os
import glob
from quench_library import compute_rho_list_umbrella_2d

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
target_kbt = target_temp * kb

for run_temp in run_temp_list:
    run_kbt = run_temp * kb
    in_dir_prefix = os.path.join(os.getcwd(),"high_T_umbrella","T%.1f"%(run_temp))
    out_dir_prefix = os.path.join(os.getcwd(),"analysis","T%.1f"%(run_temp))
    if not os.path.exists(out_dir_prefix):
        os.makedirs(out_dir_prefix,exist_ok=True)
    for phi in phi_centers:
        for psi in psi_centers:
            in_dir = os.path.join(in_dir_prefix,"phi%.2f_psi%.2f_k%.1f"%(phi,psi,kappa))
            out_dir = os.path.join(out_dir_prefix,"phi%.2f_psi%.2f_k%.1f"%(phi,psi,kappa))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir,exist_ok=True)
            print("phi%.1f,psi%.1f,rt%.1f,tt%.1f"%(phi,psi,run_temp,target_temp))
            final_file = os.path.join(out_dir,"rho_list_umbrella_phi%.2f_psi%.2f_k%.1f_tt%.1f_%dx%d_Nstep1000.npy"%(phi,psi,kappa,target_temp,fes_phi_windows,fes_psi_windows))
            if os.path.exists(final_file):
                print("final file %s exists!"%(final_file))
                continue
            restart_files = glob.glob(os.path.join(in_dir,"alanine_langevin_rt%.1f_rg%.2e_eqsteps%d_step%d.restart"%(run_temp,run_gamma,eq_steps,run_steps)))
            if len(restart_files) == 0:
                print("No valid restart files!")
                continue
            for restart_file in restart_files: # only one log file
                log_file = restart_file.replace("_step%d.restart"%(run_steps),".log")
                rho_list = compute_rho_list_umbrella_2d(log_file,fes_phi_windows,fes_psi_windows,run_kbt,target_kbt)[1:,:,:] # size (2000000, 100, 100)
                #print(rho_list.shape)
                # save first 2000 or every 1000
                np.save(os.path.join(out_dir,"rho_list_umbrella_phi%.2f_psi%.2f_k%.1f_tt%.1f_%dx%d_N2000.npy"%(phi,psi,kappa,target_temp,fes_phi_windows,fes_psi_windows)),rho_list[:2000,:,:])
                rho = np.mean((rho_list).reshape(1000,2000,fes_phi_windows,fes_psi_windows),axis=0)
                np.save(os.path.join(out_dir,"rho_list_umbrella_phi%.2f_psi%.2f_k%.1f_tt%.1f_%dx%d_Nstep1000.npy"%(phi,psi,kappa,target_temp,fes_phi_windows,fes_psi_windows)),rho)


