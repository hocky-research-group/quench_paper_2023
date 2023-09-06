def run_langevin_umbrella_alanine(command_file,input_file,out_prefix,run_gamma,run_temp,eq_steps,thermo_freq,run_steps,dt=1.0,nrestart=1,plumed_file=None,pull_plumed_file=None):
    """
        run langevin dynamics w/o umbrella sampling, damp is the inverse of run_gamma including mass
    """
    import lammps
    log_file = out_prefix + "langevin_rt%.1f_rg%.2e_eqsteps%d.log"%(run_temp,run_gamma,eq_steps)
    lmp = lammps.lammps()
    commands = open(command_file,'r').readlines()
    if input_file.split(".")[-1] == "restart":
        commands = [l.replace("__INPUT__","read_restart %s"%(input_file)) for l in commands]
    else:
        commands = [l.replace("__INPUT__","read_data %s"%(input_file)) for l in commands]
    lmp.command("log %s"%(log_file))
    for command in commands:
        lmp.command("%s"%(command.strip()))
    lmp.command("timestep %f"%(dt))
    lmp.command("thermo %d"%(int(eq_steps/100)))
    lmp.command("reset_timestep 0")
    # langevin
    damp = 1.0 / run_gamma
    lmp.command("fix 1 all langevin %.1f %.1f %f %d"%(run_temp,run_temp,damp,58728))
    lmp.command("fix 2 all nve")
    # umbrella sampling pull
    if pull_plumed_file is not None:
        lmp.command("fix 30 all plumed plumedfile %s"%(pull_plumed_file))
        lmp.command("run %d"%(eq_steps))
        lmp.command("unfix 30")
        lmp.command("reset_timestep 0")
    # umbrella sampling
    if plumed_file is not None:
        lmp.command("fix 20 all plumed plumedfile %s"%(plumed_file))
        lmp.command("run %d"%(eq_steps))
        lmp.command("reset_timestep 0")
    lmp.command("thermo %d"%(thermo_freq))
    restart_file = log_file.replace(".log","_step*.restart")
    lmp.command("restart %d %s"%(int(run_steps/nrestart),restart_file))
    lmp.command("run %d"%(run_steps))
    return log_file

def get_thermo_data(log_file):
    import lammps
    return lammps.get_thermo_data(open(log_file,'r').read())[-1].thermo

def make_plumed_file(plumed_template,out_dir,phi,psi,phi_kappa,psi_kappa,out_prefix=""):
    """ 
        replace KEY WORD in plumed template to create a specific plumed file
    """
    import os
    if not os.path.exists(plumed_template):
        print("File %s does not exist!"%(plumed_template))
        raise Exception("File not exist")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plumed_file = os.path.join(out_dir,out_prefix + os.path.basename(plumed_template))
    if os.path.exists(plumed_file):
        return plumed_file
    replace_dict = { 
        "PHI_KAPPA": "%.1f"%(phi_kappa),
        "PSI_KAPPA": "%.1f"%(psi_kappa),
        "PHI_CENTER": "%.4f"%(phi),
        "PSI_CENTER": "%.4f"%(psi),
    }   
    with open(plumed_template,'r') as plumed_template_f:
        plumed_template_text = plumed_template_f.read()
    plumed_file_f = open(plumed_file,'w')
    plumed_file_f.write(plumed_template_text%replace_dict)
    plumed_file_f.close()
    return plumed_file

def angle_distance2_pbc(angle,angle0):
    import numpy as np
    distance = angle - angle0
    while(distance > np.pi):
        distance -= 2.0 * np.pi
    while(distance < -np.pi):
        distance += 2.0 * np.pi
    return distance**2

def angle_distance2_trj_pbc(angle_trj,angle0):
    import numpy as np
    distance2_trj = np.zeros_like(angle_trj)
    for i,angle in enumerate(angle_trj):
        distance = angle - angle0
        while(distance > np.pi):
            distance -= 2.0 * np.pi
        while(distance < -np.pi):
            distance += 2.0 * np.pi
        distance2_trj[i] = distance**2
    return distance2_trj

def log_sum(exp_part_list):
    import numpy as np
    if len(exp_part_list) == 0:
        return -np.inf
    exp_part_list = np.sort(exp_part_list)[-1::-1]
    result = exp_part_list[0].copy()
    for exp_part in exp_part_list[1:]:
        if exp_part == -np.inf:
            continue
        elif result == -np.inf:
            result = exp_part.copy()
        else:
            result = result + np.log(1.0+np.exp(exp_part-result))
    return result

def compute_rho_list_umbrella_2d(log_file,fes_phi_windows,fes_psi_windows,run_kbt,target_kbt):
    import lammps
    import numpy as np
    w = (target_kbt-run_kbt)/(run_kbt*target_kbt)
    fes_dphi = np.pi * 2.0 / fes_phi_windows
    fes_dpsi = np.pi * 2.0 / fes_psi_windows
    thermo_data = lammps.get_thermo_data(open(log_file,'r').read())[-1].thermo
    phi_trj = np.array(thermo_data.c_3) * np.pi / 180.0
    psi_trj = np.array(thermo_data.c_4) * np.pi / 180.0
    etot_trj = np.array(thermo_data.TotEng)
    N = len(phi_trj)
    rho_list = np.zeros((N,fes_phi_windows,fes_psi_windows))
    for i in range(N):
        rho_list[i,int((phi_trj[i]+np.pi)/fes_dphi)%fes_phi_windows,int((psi_trj[i]+np.pi)/fes_dpsi)%fes_psi_windows] += np.exp(etot_trj[i]*w)
    return rho_list

def angle_pbc(angle_trj,angle0):
    import numpy as np
    angle_pbc_trj = np.zeros_like(angle_trj)
    for i,angle in enumerate(angle_trj):
        angle_list = np.array([angle,angle+2.0*np.pi,angle-2.0*np.pi])
        angle_pbc_trj[i] = angle_list[np.argmin(np.abs(angle_list-angle0))]
    return angle_pbc_trj

