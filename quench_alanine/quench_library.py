def run_langevin_alanine(command_file,input_file,out_prefix,run_gamma,run_temp,eq_steps,run_steps,dt=1.0,nrestart=1):
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
    lmp.command("reset_timestep 0")
    # langevin
    damp = 1.0 / run_gamma
    lmp.command("fix 1 all langevin %.1f %.1f %f %d"%(run_temp,run_temp,damp,58728))
    lmp.command("fix 2 all nve")
    lmp.command("thermo %d"%(int(eq_steps/100)))
    lmp.command("run %d"%(eq_steps))
    lmp.command("reset_timestep 0")
    lmp.command("thermo %d"%(int(run_steps/nrestart)))
    restart_file = log_file.replace(".log","_step*.restart")
    lmp.command("restart %d %s"%(int(run_steps/nrestart),restart_file))
    lmp.command("run %d"%(run_steps))
    return log_file

def run_quench_alanine(command_file,restart_file,out_dir,quench_thermo_freq,quench_gamma,gT,gT_b,dt=1.0,heat=False):
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
    # quench
    lmp.command("fix 1 all quench_exponential %f"%(quench_gamma))
    # check completion
    restart_file = log_file.replace(".log","_step*.restart")
    lmp.command("restart %d %s"%(quench_steps,restart_file))
    lmp.command("run %d"%(quench_steps))
    return log_file

def get_thermo_data(log_file):
    import lammps
    return lammps.get_thermo_data(open(log_file,'r').read())[-1].thermo

def angle_distance2_pbc(angle,angle0):
    import numpy as np
    distance = angle - angle0
    while(distance > np.pi):
        distance -= 2.0 * np.pi
    while(distance < -np.pi):
        distance += 2.0 * np.pi
    return distance**2

def log_sum(exp_part_list):
    import numpy as np
    if len(exp_part_list) == 0:
        return -np.inf
    exp_part_list = np.flip(np.sort(exp_part_list))
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

def log_sub_binary(lnA,lnB):
    import numpy as np
    return lnA + np.log(1.0 - np.exp(lnB - lnA))

def angle_pbc(angle_trj,angle0):
    import numpy as np
    angle_pbc_trj = np.zeros_like(angle_trj)
    for i,angle in enumerate(angle_trj):
        angle_list = np.array([angle,angle+2.0*np.pi,angle-2.0*np.pi])
        angle_pbc_trj[i] = angle_list[np.argmin(np.abs(angle_list-angle0))]
    return angle_pbc_trj

def find_Ebound(log_file,index,heat=True):
    Emin = get_thermo_data(log_file).TotEng[-1]
    if heat:
        Emax = get_thermo_data(log_file.replace("qg","qg-")).TotEng[-1]
    else:
        Emax = get_thermo_data(log_file).TotEng[0]
    return index,Emin,Emax

def infinite_stopping_compute_lnrho_2d(log_file,fes_phi_windows,fes_psi_windows,target_kbt,run_kbt,dof,dt,quench_gamma,Emin,Emax,index,heat=True):
    fes_dphi = np.pi * 2.0 / fes_phi_windows
    fes_dpsi = np.pi * 2.0 / fes_psi_windows
    # merge
    thermo_data = get_thermo_data(log_file)
    etot_trj = np.array(thermo_data.TotEng)
    phi_trj = np.array(thermo_data.c_3) * np.pi / 180.0
    psi_trj = np.array(thermo_data.c_4) * np.pi / 180.0
    t_trj = np.array(thermo_data.Step)
    if heat:
        log_b_file = log_file.replace("qg","qg-")
        thermo_data_b = get_thermo_data(log_b_file)
        etot_trj_b = np.array(thermo_data_b.TotEng)[1:]
        origin = len(etot_trj_b)
        etot_trj = np.append(np.flip(etot_trj_b),etot_trj)
        phi_trj_b = np.array(thermo_data_b.c_3)[1:] * np.pi / 180.0
        phi_trj = np.append(np.flip(phi_trj_b),phi_trj)
        psi_trj_b = np.array(thermo_data_b.c_4)[1:] * np.pi / 180.0
        psi_trj = np.append(np.flip(psi_trj_b),psi_trj)
        t_trj_b = np.array(thermo_data_b.Step)[1:] * -0.2
        t_trj = np.append(np.flip(t_trj_b),t_trj) * dt
    dgt_trj = t_trj * dof * quench_gamma
    ln_numerator_trj = -etot_trj / target_kbt - dgt_trj
    ln_denominator_trj = -etot_trj / run_kbt - dgt_trj
    # find tau^- and tau^+
    Emin_ind = np.argwhere(etot_trj < Emin)
    Emax_ind = np.argwhere(etot_trj > Emax)
    if len(Emax_ind) == 0:
        start = 0
    else:
        start = np.max(Emax_ind) + 1
    if len(Emin_ind) == 0:
        end = len(etot_trj) - 1
    else:
        end = np.min(Emin_ind) - 1
    ln_denominator = log_sum(ln_denominator_trj[start:end+1])
    lnrho = np.ones((fes_phi_windows,fes_psi_windows))*-np.inf
    for t in range(start,end+1):
        k = int((phi_trj[t]+np.pi)/fes_dphi)%fes_phi_windows
        l = int((psi_trj[t]+np.pi)/fes_dpsi)%fes_psi_windows
        lnrho[k,l] = log_sum_binary(lnrho[k,l],ln_numerator_trj[t])
    lnrho -= ln_denominator
    lnQ = log_sum(ln_numerator_trj[start:end+1]) - ln_denominator
    time = t_trj[end] - t_trj[start]
    return index,lnrho,lnQ,time

