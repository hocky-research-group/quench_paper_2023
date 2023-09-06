import numpy as np
import os
import glob
from quench_library import log_sum

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
dof = 3.0 * N

in_dir_prefix = os.path.join(os.getcwd(),"analysis")
out_dir = os.path.join(os.getcwd(),"analysis")
for quench_gamma in quench_gamma_list:
    error_list = np.zeros(len(run_temp_list))
    for i,run_temp in enumerate(run_temp_list):
        ideal_value = np.log(target_temp / run_temp) * dof
        in_dir = os.path.join(in_dir_prefix,"N%d"%(N),"T%.1f"%(run_temp))
        lnQ = np.load(os.path.join(in_dir,"infinite_stopping_lnQ_N%d_K%.1f_rt%.1f_rg%.2e_qg%.2e.npy"%(N,kappa,run_temp,run_gamma,quench_gamma)))
        lnQ -= np.log(nrestarts)
        error_list[i] = np.abs((lnQ-ideal_value) / ideal_value)
    print(error_list)
    np.save(os.path.join(out_dir,"infinite_stopping_error_N%d_qg%.2e_T.npy"%(N,quench_gamma)),error_list)


