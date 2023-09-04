def run_quench_umbrella_alanine(command_file,restart_file,out_dir,quench_thermo_freq,quench_gamma,gT,gT_b,dt=1.0,heat=False,plumed_file=None):
    """
        run quench simulations w/o umbrella sampling
    """
    import os
    import lammps
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    if heat:
        quench_steps = int(gT_b / -quench_gamma / dt)
    else:
        quench_steps = int(gT / quench_gamma / dt)
    lmp = lammps.lammps()
    log_file = os.path.join(out_dir,os.path.basename(restart_file).replace(".restart","_gT%d_gTb%d_qg%.2e.log"%(gT,gT_b,quench_gamma)))
    commands = open(command_file).readlines()
    commands = [l.replace("__INPUT__","read_restart %s"%(restart_file)) for l in commands]
    lmp.command("log %s"%(log_file))
    for command in commands:
        lmp.command(command.strip())
    lmp.command("reset_timestep 0")
    lmp.command("timestep %f"%(dt))
    if heat:
        lmp.command("variable vx atom -vx")
        lmp.command("variable vy atom -vy")
        lmp.command("variable vz atom -vz")
        lmp.command("velocity all set v_vx v_vy v_vz")
    lmp.command("thermo %d"%(quench_thermo_freq))
    # umbrella sampling
    if plumed_file is not None:
        lmp.command("fix 20 all plumed plumedfile %s"%(plumed_file))
        lmp.command("fix_modify 20 energy yes") # add biased potential in thermo print
    # quench
    lmp.command("fix 1 ALA quench_exponential %f"%(quench_gamma))
    lmp.command("fix 2 SOL nve")
    # check completion
    restart_file = log_file.replace(".log","_step*.restart")
    lmp.command("restart %d %s"%(quench_steps,restart_file))
    lmp.command("run %d"%(quench_steps))
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

def log_sum_binary(lnA,lnB):
    import numpy as np
    if lnA == -np.inf:
        return lnB
    if lnB == -np.inf:
        return lnA
    if lnA >= lnB:
        return lnA + np.log(1.0+np.exp(lnB-lnA))
    else:
        return lnB + np.log(1.0+np.exp(lnA-lnB))

def compute_lnrho_2d(log_file,fes_phi_windows,fes_psi_windows,target_kbt,run_kbt,dof,dt,quench_gamma,index,heat=False):
    import lammps
    import numpy as np
    fes_dphi = np.pi * 2.0 / fes_phi_windows
    fes_dpsi = np.pi * 2.0 / fes_psi_windows
    thermo_data = lammps.get_thermo_data(open(log_file,'r').read())[-1].thermo
    e_trj = np.array(thermo_data.TotEng)
    t_trj = np.array(thermo_data.Step)
    phi_trj = np.array(thermo_data.c_3) * np.pi / 180.0
    psi_trj = np.array(thermo_data.c_4) * np.pi / 180.0
    if heat:
        log_b_file = log_file.replace("qg","qg-")
        thermo_data = lammps.get_thermo_data(open(log_b_file,'r').read())[-1].thermo
        e_trj = np.append(e_trj,np.array(thermo_data.TotEng)[1:])
        t_trj = np.append(t_trj,0.1*-np.array(thermo_data.Step)[1:]) # dt=0.1
        phi_trj = np.append(phi_trj,np.array(thermo_data.c_3)[1:] * np.pi / 180.0)
        psi_trj = np.append(psi_trj,np.array(thermo_data.c_4)[1:] * np.pi / 180.0)
    dgt_trj = t_trj * dt * quench_gamma * dof
    numerator_exp_part_list = -e_trj/target_kbt - dgt_trj
    ln_numerator = log_sum(numerator_exp_part_list)
    L = len(e_trj)
    lnrho = np.ones((fes_phi_windows,fes_psi_windows))*-np.inf
    for t in range(L):
        k = int((phi_trj[t]+np.pi)/fes_dphi)%fes_phi_windows
        l = int((psi_trj[t]+np.pi)/fes_dpsi)%fes_psi_windows
        lnrho[k,l] = log_sum_binary(lnrho[k,l],numerator_exp_part_list[t])
    lnrho -= ln_numerator
    return index,lnrho

def compute_N_2d(log_file,fes_phi_windows,fes_psi_windows,run_kbt,target_kbt,index,heat=False):
    import lammps
    import numpy as np
    w = (target_kbt-run_kbt) / (run_kbt*target_kbt)
    fes_dphi = np.pi * 2.0 / fes_phi_windows
    fes_dpsi = np.pi * 2.0 / fes_psi_windows
    thermo_data = lammps.get_thermo_data(open(log_file,'r').read())[-1].thermo
    e_trj = np.array(thermo_data.TotEng)
    phi_trj = np.array(thermo_data.c_3) * np.pi / 180.0
    psi_trj = np.array(thermo_data.c_4) * np.pi / 180.0
    if heat:
        log_b_file = log_file.replace("qg","qg-")
        thermo_data = lammps.get_thermo_data(open(log_b_file,'r').read())[-1].thermo
        e_trj = np.append(e_trj,np.array(thermo_data.TotEng)[1:])
        phi_trj = np.append(phi_trj,np.array(thermo_data.c_3)[1:] * np.pi / 180.0)
        psi_trj = np.append(psi_trj,np.array(thermo_data.c_4)[1:] * np.pi / 180.0)
    N_kl = np.zeros((fes_phi_windows,fes_psi_windows))
    for i in range(len(phi_trj)):
        N_kl[int((phi_trj[i]+np.pi)/fes_dphi)%fes_phi_windows,int((psi_trj[i]+np.pi)/fes_dpsi)%fes_psi_windows] += np.exp(w*e_trj[i])
    return index,N_kl

def angle_pbc(angle_trj,angle0):
    import numpy as np
    angle_pbc_trj = np.zeros_like(angle_trj)
    for i,angle in enumerate(angle_trj):
        angle_list = np.array([angle,angle+2.0*np.pi,angle-2.0*np.pi])
        angle_pbc_trj[i] = angle_list[np.argmin(np.abs(angle_list-angle0))]
    return angle_pbc_trj

