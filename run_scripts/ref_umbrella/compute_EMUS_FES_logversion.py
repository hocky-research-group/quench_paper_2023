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

for run_temp in run_temp_list:
    target_temp = run_temp
    target_kbt = target_temp * kb
    run_kbt = run_temp * kb
    in_dir = os.path.join(os.getcwd(),"EMUS","T%.1f"%(run_temp))
    rho_matrix = np.exp(np.load(os.path.join(in_dir,"lnrho_matrix.npy"))) / 2000001
    print("shape of rho matrix:",rho_matrix.shape)
    one_matrix = np.exp(np.load(os.path.join(in_dir,"lnone_matrix.npy"))) / 2000001
    print("shape of one matrix:",one_matrix.shape)
    F_matrix = np.exp(np.load(os.path.join(in_dir,"lnF_matrix.npy"))) / 2000001
    print("shape of F matrix:",F_matrix.shape)
    print(np.sum(F_matrix,axis=1))
    A = np.eye(phi_windows*psi_windows) - F_matrix
    Q,R = np.linalg.qr(A,mode='complete')
    print("Q shape,R shape:",Q.shape,R.shape)
    z = Q[:,-1]
    print("sum(zF-z):",np.sum(np.matmul(z,F_matrix)-z))
    z /= np.sum(z)
    rho = np.zeros((fes_phi_windows,fes_psi_windows))
    for k in range(fes_phi_windows):
        for l in range(fes_psi_windows):
            rho[k,l] = np.sum(z*rho_matrix[:,k,l]) / np.sum(z*one_matrix)
    np.save(os.path.join(in_dir,"rho_EMUS_logversion.npy"),rho)
    print("shape of rho:",rho.shape)
    F = -target_kbt * np.log(rho)
    F -= F.min()
    print(F)
    np.save(os.path.join(in_dir,"F_EMUS_rt%.1f_tt%.1f.npy"%(run_temp,run_temp)),F)

