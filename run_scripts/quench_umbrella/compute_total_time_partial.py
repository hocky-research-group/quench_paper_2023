import numpy as np
import os
import glob
from quench_library import angle_distance2_pbc,log_sum

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
target_temp_list = [300.0]
quench_gamma_list = [0.001]

for target_temp in target_temp_list:
    run_temp = target_temp
    run_kbt = run_temp * kb
    target_kbt = target_temp * kb
    in_dir_prefix = os.path.join(os.getcwd(),"analysis","T%.1f"%(run_temp))
    out_dir = os.path.join(os.getcwd(),"wham_analysis","T%.1f"%(run_temp),"rho%dx%d_k%.1f"%(phi_windows,psi_windows,kappa))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for quench_gamma in quench_gamma_list:
        #num_restart_list = [1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
        num_restart_list = [333,666]
        for num_restart in num_restart_list:
            time_file = os.path.join(out_dir,"infinite_stopping_time_%dx%d_%dx%d_k%.1f_tt%.1f_qg%.2e_N%d.npy"%(phi_windows,psi_windows,fes_phi_windows,fes_psi_windows,kappa,target_temp,quench_gamma,num_restart))
            #if not os.path.exists(time_file):
            if True:
                time_ij = np.zeros((phi_windows,psi_windows))
                for i,phi in enumerate(phi_centers):
                    for j,psi in enumerate(psi_centers):
                        in_dir = os.path.join(in_dir_prefix,"phi%.2f_psi%.2f_k%.1f"%(phi,psi,kappa))
                        time_ij_list = np.load(os.path.join(in_dir,"infinite_stopping_time_list_psi%.2f_k%.1f_qg%.2e_tt%.1f_%dx%d.npy"%(psi,kappa,quench_gamma,target_temp,fes_phi_windows,fes_psi_windows))) + 1000.0
                        time_ij_list = time_ij_list[:num_restart]
                        time_ij[i,j] = np.sum(time_ij_list,axis=0)
                np.save(os.path.join(out_dir,"infinite_stopping_time_%dx%d_%dx%d_k%.1f_tt%.1f_qg%.2e_N%d.npy"%(phi_windows,psi_windows,fes_phi_windows,fes_psi_windows,kappa,target_temp,quench_gamma,num_restart)),np.sum(time_ij))
            print(np.load(time_file))


