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
run_kbt = run_temp * kb
target_kbt = target_temp * kb

in_dir_prefix = os.path.join(os.getcwd(),"analysis","T%.1f"%(run_temp))
out_dir = os.path.join(os.getcwd(),"wham_analysis","T%.1f"%(run_temp),"rho%dx%d_k%.1f"%(phi_windows,psi_windows,kappa))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# generate w_ij_kl
w_file = os.path.join(out_dir,"w_%dx%d_%dx%d_k%.1f.npy"%(phi_windows,psi_windows,fes_phi_windows,fes_psi_windows,kappa))
if not os.path.exists(w_file):
    w_ij_kl = np.zeros((phi_windows,psi_windows,fes_phi_windows,fes_psi_windows))
    for i,phi in enumerate(phi_centers):
        for j,psi in enumerate(psi_centers):
            for k,fes_phi in enumerate(fes_phi_centers):
                phi_distance2 = angle_distance2_pbc(phi,fes_phi)
                for l,fes_psi in enumerate(fes_psi_centers):
                    psi_distance2 = angle_distance2_pbc(psi,fes_psi)
                    w_ij_kl[i,j,k,l] = 0.5 * kappa * (phi_distance2 + psi_distance2)
    np.save(os.path.join(out_dir,"w_%dx%d_%dx%d_k%.1f.npy"%(phi_windows,psi_windows,fes_phi_windows,fes_psi_windows,kappa)),w_ij_kl)
    print("shape",w_ij_kl.shape)

for quench_gamma in quench_gamma_list:
    for num_restart in num_restart_list:
        # generate rho_ij_kl
        rho_file = os.path.join(out_dir,"N_%dx%d_%dx%d_k%.1f_tt%.1f_qg%.2e_N%d.npy"%(phi_windows,psi_windows,fes_phi_windows,fes_psi_windows,kappa,target_temp,quench_gamma,num_restart))
        if not os.path.exists(rho_file):
            N_ij_kl = np.zeros((phi_windows,psi_windows,fes_phi_windows,fes_psi_windows))
            for i,phi in enumerate(phi_centers):
                for j,psi in enumerate(psi_centers):
                    in_dir = os.path.join(in_dir_prefix,"phi%.2f_psi%.2f_k%.1f"%(phi,psi,kappa))
                    N_ij_list = np.load(os.path.join(in_dir,"N_list_phi%.2f_psi%.2f_k%.1f_qg%.2e_tt%.1f_%dx%d.npy"%(phi,psi,kappa,quench_gamma,target_temp,fes_phi_windows,fes_psi_windows)))
                    N_ij_list = N_ij_list[:num_restart,:,:]
                    N_ij = np.zeros((fes_phi_windows,fes_psi_windows))
                    for k in range(fes_phi_windows):
                        for l in range(fes_psi_windows):
                            N_ij[k,l] = np.sum(N_ij_list[:,k,l])
                    # normalize
                    N_ij_kl[i,j,:,:] = N_ij
            np.save(os.path.join(out_dir,"N_%dx%d_%dx%d_k%.1f_tt%.1f_qg%.2e_N%d.npy"%(phi_windows,psi_windows,fes_phi_windows,fes_psi_windows,kappa,target_temp,quench_gamma,num_restart)),N_ij_kl)
            print("shape",N_ij_kl.shape)
        
        # initialize
        rho_ij_kl = np.load(os.path.join(out_dir,"N_%dx%d_%dx%d_k%.1f_tt%.1f_qg%.2e_N%d.npy"%(phi_windows,psi_windows,fes_phi_windows,fes_psi_windows,kappa,target_temp,quench_gamma,num_restart)))
        w_ij_kl = np.load(os.path.join(out_dir,"w_%dx%d_%dx%d_k%.1f.npy"%(phi_windows,psi_windows,fes_phi_windows,fes_psi_windows,kappa)))
        print("rho_ij_kl:",rho_ij_kl)
        print("w_ij_kl:",w_ij_kl)
        N_ij = np.sum(rho_ij_kl,axis=(-2,-1))
        N_kl = np.sum(rho_ij_kl,axis=(0,1))
        print("N_ij:",N_ij)
        print("N_kl:",N_kl)
        rho_kl = np.ones((fes_phi_windows,fes_psi_windows))
        F_kl = -target_kbt * np.log(rho_kl)
        F_kl -= F_kl.min()
        c_ij_inverse = np.zeros((phi_windows,psi_windows))
        for i in range(phi_windows):
            for j in range(psi_windows):
                c_ij_inverse[i,j] = np.sum(rho_kl * np.exp(-w_ij_kl[i,j,:,:]/target_kbt))
        
        # WHAM equations
        step = 0
        error = 1.0
        while(error > tolerance):
            step += 1
            old_rho_kl = rho_kl.copy()
            old_F_kl = F_kl.copy()
            old_c_ij_inverse = c_ij_inverse.copy()
            # update rho_kl,F_kl, minimize F_kl to 0
            for k in range(fes_phi_windows):
                for l in range(fes_psi_windows):
                    rho_kl[k,l] = N_kl[k,l] / np.sum(N_ij * np.exp(-w_ij_kl[:,:,k,l]/target_kbt) / old_c_ij_inverse)
            mask = rho_kl != 0.
            old_mask = old_rho_kl != 0.
            F_kl = -target_kbt * np.log(rho_kl)
            F_kl -= F_kl.min()
            # update c_ij_inverse
            for i in range(phi_windows):
                for j in range(psi_windows):
                    c_ij_inverse[i,j] = np.sum(rho_kl * np.exp(-w_ij_kl[i,j,:,:]/target_kbt))
            # compute error of F_kl (largest)
            total_mask = np.logical_and(mask,old_mask)
            error = np.abs(F_kl[total_mask] - old_F_kl[total_mask]).max()
            print("Step %d, error = %f"%(step,error))
        
        # save results
        np.save(os.path.join(out_dir,"solv_rho_%dx%d_k%.1f_rt%.1f_tt%.1f_qg%.2e_N%d_N.npy"%(fes_phi_windows,fes_psi_windows,kappa,run_temp,target_temp,quench_gamma,num_restart)),rho_kl)
        np.save(os.path.join(out_dir,"solv_F_%dx%d_k%.1f_rt%.1f_tt%.1f_qg%.2e_N%d_N.npy"%(fes_phi_windows,fes_psi_windows,kappa,run_temp,target_temp,quench_gamma,num_restart)),F_kl)
        rho_kl = np.load(os.path.join(out_dir,"solv_rho_%dx%d_k%.1f_rt%.1f_tt%.1f_qg%.2e_N%d_N.npy"%(fes_phi_windows,fes_psi_windows,kappa,run_temp,target_temp,quench_gamma,num_restart)))
        F_kl = np.load(os.path.join(out_dir,"solv_F_%dx%d_k%.1f_rt%.1f_tt%.1f_qg%.2e_N%d_N.npy"%(fes_phi_windows,fes_psi_windows,kappa,run_temp,target_temp,quench_gamma,num_restart)))
        print(F_kl)
 
