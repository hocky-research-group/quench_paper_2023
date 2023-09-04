from quench_library import run_langevin_alanine
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
    globals().update(high_T_params)

result_list = []
out_dir = os.path.join(os.getcwd(),"high_T","T%.1f"%(run_temp))
if not os.path.exists(out_dir):
    os.makedirs(out_dir,exist_ok=True)
out_prefix = os.path.join(out_dir,"alanine_")
r = run_langevin_alanine(command_file,input_file,out_prefix,run_gamma,run_temp,eq_steps,run_steps,dt=dt,nrestart=nrestart)
result_list.append(r)

for r in result_list:
    try:
        result = r
        print(f"Got results: {result}.")
    except Exception as e:
        print(f"Application {r} failed but continuing.")

